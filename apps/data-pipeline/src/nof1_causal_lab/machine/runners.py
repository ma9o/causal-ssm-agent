"""Transition execution against the versioned artifact store.

Each runner receives explicit input pins selected by the machine before
execution. It must read exactly those versions, write new artifact versions with
the same pins in ``derived_from``, and return effects for the workflow to apply.
Heavy transitions can be routed to Modal, but routing is infra-only: it cannot
change the pinned versions or the derivation cascade applied to the result.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename, pickle_filename
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import (
    ExecOptions,
    TransitionEffects,
    input_pins,
    run_retractions,
)
from nof1_causal_lab.machine.store import ArtifactStore

if TYPE_CHECKING:
    import polars as pl
    from pydantic import BaseModel

    from nof1_causal_lab.machine.artifacts import ArtifactId, ArtifactVersionInfo, EpisodeState


def _filter_to_contract(cls: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    fields = set(cls.model_fields.keys())
    return {key: value for key, value in data.items() if key in fields}


def _panel_df(store: ArtifactStore, pins: dict[ArtifactId, int]) -> pl.DataFrame:
    return store.read_parquet_file("panel", pins["panel"], parquet_filename("panel", "panel"))


async def _run_posterior(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> list[ArtifactVersionInfo]:
    from nof1_causal_lab.flows.artifact_contracts import PosteriorContract
    from nof1_causal_lab.flows.transitions.inference.flow import (
        build_sampler_config,
        run_inference_with_data,
    )
    from nof1_causal_lab.models.causal_proofs import CausalDesignRef, PosteriorProvenance
    from nof1_causal_lab.models.ssm.compile.contracts import CompiledSSMArtifact
    from nof1_causal_lab.utils.config import get_config

    compiled_ssm = CompiledSSMArtifact.model_validate(
        store.read_json_file(
            "compiled_ssm", pins["compiled_ssm"], json_filename("compiled_ssm", "compiled_ssm")
        )
    )
    panel = _panel_df(store, pins)
    compiled_meta = store.read_meta("compiled_ssm", pins["compiled_ssm"])
    provenance = PosteriorProvenance(
        causal_design=CausalDesignRef(
            workspace_id=workspace_id,
            version=compiled_meta.derived_from["causal_design"],
        ),
        compiled_ssm_version=pins["compiled_ssm"],
        panel_version=pins["panel"],
    )

    result = await asyncio.to_thread(
        run_inference_with_data,
        compiled_ssm=compiled_ssm,
        data_for_model=panel,
        sampler_config=build_sampler_config(options.inference_method),
        provenance=provenance,
        workspace_id=workspace_id,
        compute_loo_diagnostics=get_config().inference.compute_loo_diagnostics,
    )

    fitted_artifact = result.pop("_fitted_artifact", None)
    payload = _filter_to_contract(PosteriorContract, result)
    info = store.write_version(
        "posterior",
        provenance="computed",
        derived_from=pins,
        produced_by="run:posterior",
        json_files={json_filename("posterior", "diagnostics"): payload},
        pickle_files={pickle_filename("posterior", "fitted"): fitted_artifact},
    )
    return [info]


_TRANSITION_RUNNERS = {
    "posterior": _run_posterior,
}

_TEMPORAL_ONLY_TRANSITIONS = frozenset(
    {
        "raw_data",
        "latent_structure",
        "measurement_structure",
        "measurements",
        "statistical_model_spec",
        "baseline_report",
    }
)

_MODAL_TRANSITIONS = frozenset({"posterior"})


async def execute_transition_locally(
    workspace_id: str,
    artifact_id: ArtifactId,
    pins: dict[ArtifactId, int],
    state: EpisodeState,
    options: ExecOptions,
) -> TransitionEffects:
    """Run a transition on this process against pinned input versions."""
    from nof1_causal_lab.flows.runtime_events import emit_transition_event

    store = ArtifactStore(workspace_id)
    if artifact_id in _TEMPORAL_ONLY_TRANSITIONS:
        raise RuntimeError(f"{artifact_id} is implemented only as a Temporal child workflow")
    runner = _TRANSITION_RUNNERS[artifact_id]
    spec = transition_spec(artifact_id)
    emit_transition_event(workspace_id, artifact_id, "running")
    try:
        produced = await runner(workspace_id, store, pins, options)
        retracted = run_retractions(state, spec, produced)
        effects = complete_derivation_cascade(store, state, produced, retracted)
    except Exception as exc:
        emit_transition_event(
            workspace_id,
            artifact_id,
            "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    emit_transition_event(workspace_id, artifact_id, "completed")
    return effects


async def execute_transition(
    workspace_id: str,
    artifact_id: ArtifactId,
    state: EpisodeState,
    options: ExecOptions,
) -> TransitionEffects:
    """Run a transition, routing heavy transitions to Modal in production."""
    spec = transition_spec(artifact_id)
    pins = input_pins(state, spec)
    if artifact_id in _TEMPORAL_ONLY_TRANSITIONS:
        raise RuntimeError(f"{artifact_id} is implemented only as a Temporal child workflow")
    if os.environ.get("DEPLOYMENT_ENV") == "production" and artifact_id in _MODAL_TRANSITIONS:
        from nof1_causal_lab.flows.modal_runners import run_transition_on_modal

        return await run_transition_on_modal(workspace_id, artifact_id, pins, state, options)
    return await execute_transition_locally(workspace_id, artifact_id, pins, state, options)
