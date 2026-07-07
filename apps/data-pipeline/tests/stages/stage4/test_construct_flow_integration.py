"""Integration regression: drive the whole construct-build loop on real data.

A scripted (LLM-free) session submits one construct at a time; the loop runs the
exact prior-predictive reachability battery over real observation rows (via
``prepare_model_runtime``) and, on success, assembles a StatisticalModelSpec + priors that
compile to a live ``compiled_ssm``. This is the offline proof that the loop the
stage flow drives actually works end to end before the real LLM is pointed at it.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nof1_causal_lab.flows.stages.stage4.agentic.stage4_construct_flow import (
    run_stage4_construct_build,
)
from tests.models.ssm.test_dag_to_ssm import _make_causal_design_dict

# Every soft check is pre-accepted so admission turns only on the hard checks
# (finite simulation + reachable data location) — the reachability *values* of
# small random fixtures are not the point here; the plumbing is.
_ACCEPT_ALL_SOFT = {
    "C1b confinement": "fixture",
    "C2 latent scale": "fixture",
    "C3 resolvability": "fixture",
    "C4b edge overwhelm": "fixture",
    "C4c saturation": "fixture",
    "C5b width": "fixture",
    "C5c transmission": "fixture",
}


def _normal(mu: float, sigma: float) -> dict:
    return {"distribution": "Normal", "params": {"mu": mu, "sigma": sigma}}


def _halfnormal(sigma: float) -> dict:
    return {"distribution": "HalfNormal", "params": {"sigma": sigma}}


def _indicator_rows(name: str, values: np.ndarray, anchor_times: list[str]) -> dict:
    n = len(values)
    return {
        "indicator": [name] * n,
        "value": values.tolist(),
        "anchor_time": anchor_times,
        "support_start": anchor_times,
        "support_end": anchor_times,
        "support_kind": ["point"] * n,
        "summary_operator": ["last"] * n,
        "anchor_policy": ["support_end"] * n,
        "observation_window": [None] * n,
    }


def _real_data_for_model() -> pl.DataFrame:
    """Standardized daily observations for indicators x1, x2, y1, z1 (X→Y→Z)."""
    rng = np.random.default_rng(7)
    n = 40
    anchor_times = (
        pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%dT00:00:00Z").tolist()
    )
    frames = [
        _indicator_rows(name, rng.standard_normal(n).astype(float), anchor_times)
        for name in ("x1", "x2", "y1", "z1")
    ]
    merged = {key: [v for f in frames for v in f[key]] for key in frames[0]}
    return pl.DataFrame(merged)


def _script() -> dict[str, dict]:
    return {
        "X": {
            "construct": "X",
            "indicators": [
                {"variable": "x1", "family": "gaussian", "link": "identity"},
                {"variable": "x2", "family": "gaussian", "link": "identity"},
            ],
            "priors": {
                "rho_X": _normal(0.6, 0.1),
                "sigma_X": _halfnormal(0.5),
                "lambda_x2_X": _normal(1.0, 0.2),
                "obs_sd_x1": _halfnormal(0.5),
                "obs_sd_x2": _halfnormal(0.5),
            },
            "accept": _ACCEPT_ALL_SOFT,
        },
        "Y": {
            "construct": "Y",
            "indicators": [{"variable": "y1", "family": "gaussian", "link": "identity"}],
            "priors": {
                "rho_Y": _normal(0.6, 0.1),
                "sigma_Y": _halfnormal(0.5),
                "beta_X_Y": _normal(0.3, 0.1),
            },
            "accept": _ACCEPT_ALL_SOFT,
        },
        "Z": {
            "construct": "Z",
            "indicators": [{"variable": "z1", "family": "gaussian", "link": "identity"}],
            "priors": {
                "rho_Z": _normal(0.6, 0.1),
                "sigma_Z": _halfnormal(0.5),
                "beta_Y_Z": _normal(0.3, 0.1),
            },
            "accept": _ACCEPT_ALL_SOFT,
        },
    }


class _ScriptedSession:
    """Stands in for an LLM AgentSession: calls submit_construct with the script."""

    def __init__(self, tools: list, script: dict[str, dict]) -> None:
        self._tools = {t.name: t for t in tools}
        self._script = script

    async def __aenter__(self) -> _ScriptedSession:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def turn(self, user_msg: str) -> None:
        match = re.search(r"Active construct: `([^`]+)`", user_msg)
        assert match, "prompt did not name the active construct"
        payload = self._script[match.group(1)]
        await self._tools["submit_construct"].execute(**payload)


class _ScriptedFactory:
    def __init__(self, script: dict[str, dict]) -> None:
        self._script = script

    def open(self, *, system_prompt: str, tools: list, log_label: str) -> _ScriptedSession:
        del system_prompt, log_label
        return _ScriptedSession(tools, self._script)


@pytest.mark.slow
def test_construct_build_over_real_data_compiles():
    causal_design = _make_causal_design_dict()
    result = asyncio.run(
        run_stage4_construct_build(
            causal_design=causal_design,
            question="Does X drive Y drive Z?",
            data_for_model=_real_data_for_model(),
            indicator_audits={},
            session_factory=_ScriptedFactory(_script()),
            n_draws=48,
        )
    )
    # The accumulated spec + priors are the exact inputs the stage materializes.
    assert set(result.authored_priors) >= {"rho_X", "beta_X_Y", "beta_Y_Z"}
    latent_names = [lik["variable"] for lik in result.statistical_model_spec["likelihoods"]]
    assert set(latent_names) == {"x1", "x2", "y1", "z1"}

    from nof1_causal_lab.flows.stages.stage4.assembly import materialize_stage4_result

    materialized = materialize_stage4_result(
        statistical_model_spec=result.statistical_model_spec,
        authored_priors=result.authored_priors,
        data_for_model=_real_data_for_model(),
        indicator_audits={},
        causal_design=causal_design,
        skip_ppc=True,  # the construct loop's reachability battery is the validation
    )
    assert materialized.get("_compiled_ssm") is not None
    assert materialized["statistical_model_spec"] is not None
