"""Fixture-backed integration tests for production local transition runners."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.machine.artifact_files import json_filename
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import ExecOptions, input_pins
from nof1_causal_lab.machine.runners import execute_transition_locally
from tests.artifact_contract_support import validate_artifact_payload
from tests.helpers import run_async
from tests.integration import transition_runner_fixtures as fx

if TYPE_CHECKING:
    from nof1_causal_lab.machine.store import ArtifactStore


def _run_transition(
    workspace_id: str,
    state,
    artifact_id: str,
    options: ExecOptions | None = None,
):
    spec = transition_spec(artifact_id)
    pins = input_pins(state, spec)
    return run_async(
        execute_transition_locally(
            workspace_id,
            artifact_id,
            pins,
            state,
            options or ExecOptions(),
        )
    )


def _produced(effects, artifact_id: str):
    return next(info for info in effects.produced if info.artifact_id == artifact_id)


def _assert_contract(store: ArtifactStore, artifact_id: str, version: int, context_id: str) -> dict:
    file_by_context = {
        "posterior": ("posterior", "diagnostics"),
    }[context_id]
    expected_artifact, key = file_by_context
    assert artifact_id == expected_artifact
    payload = store.read_json_file(artifact_id, version, json_filename(artifact_id, key))
    validate_artifact_payload(artifact_id, payload)
    return payload


def test_posterior_persists_posterior_from_seeded_model_artifacts(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    monkeypatch,
) -> None:
    from nof1_causal_lab.flows.transitions.inference import fit as stage5_fit

    compiled_ssm = fx.seed_compiled_ssm(artifact_store)
    panel = fx.seed_panel(artifact_store)

    def fake_fit_model(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "fitted": True,
            "inference_type": "marginal_particle_gibbs",
            "n_samples": 4,
            "duration_seconds": 0.01,
            "result": None,
            "spec": None,
            "runtime": SimpleNamespace(observation_support=None),
            "times": [0.0, 1.0],
            "mcmc_diagnostics": None,
            "smc_diagnostics": None,
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
        }

    monkeypatch.setattr(stage5_fit, "fit_model", fake_fit_model)
    monkeypatch.setattr(
        stage5_fit,
        "run_ppc",
        lambda _fitted: {
            "per_variable_warnings": [],
            "checked": True,
            "overlays": [],
            "test_stats": [],
        },
    )
    state = fx.state_from(compiled_ssm, panel)

    effects = _run_transition(integration_workspace, state, "posterior")

    info = _produced(effects, "posterior")
    assert info.derived_from == {
        "compiled_ssm": compiled_ssm.version,
        "panel": panel.version,
    }
    payload = _assert_contract(artifact_store, "posterior", info.version, "posterior")
    assert payload["inference_metadata"]["n_samples"] == 4
