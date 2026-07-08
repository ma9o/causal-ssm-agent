"""Shared fixtures + re-exports for the surviving model-spec tests.

Consumed by ``conftest.py`` (the ``simple_*`` fixtures), ``test_ssm_validation.py``
(pure builders + third-party re-exports), and ``test_grounding.py``
(``make_causal_design_dict``).
"""

# ruff: noqa: F401 — this module deliberately re-exports names for its consumers.

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nof1_causal_lab.flows.transitions.model_spec.flow import _model_spec_generate_config
from nof1_causal_lab.models.predictive_simulation import PredictiveObservationMeanOverflow
from nof1_causal_lab.models.prior_predictive import (
    get_failed_parameters,
    validate_prior_predictive,
)
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_priors as compile_ssm_priors,
)
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_ssm_inputs_from_statistical_model_spec,
)
from nof1_causal_lab.utils.openrouter_client import GenerateConfig
from nof1_causal_lab.workers.schemas_prior import PriorValidationResult


def make_causal_design_dict(
    constructs: list[dict],
    edges: list[dict],
    indicators: list[dict],
    *,
    model_clock: str | None = "1d",
) -> dict:
    """Build a CausalDesign dict (latent + measurement + estimation) for tests.

    Defaults indicator polarity to ``positive`` when not set; estimation block
    is derived from ``constructs`` (state_order = construct names) and ``edges``.
    Pass ``model_clock=None`` to omit the field entirely.
    """
    indicators = [
        {"construct_polarity": "positive", **indicator}
        if "construct_polarity" not in indicator
        else dict(indicator)
        for indicator in indicators
    ]
    measurement: dict = {"indicators": indicators}
    if model_clock is not None:
        measurement["model_clock"] = model_clock
    return {
        "latent": {"constructs": constructs, "edges": edges},
        "measurement": measurement,
        "estimation": {
            "state_order": [c["name"] for c in constructs],
            "edges": edges,
            "induced_dependencies": [],
        },
    }


def _with_positive_indicator_polarity(spec: dict[str, Any]) -> dict[str, Any]:
    """Backfill valid default indicator semantics for model-spec test fixtures."""
    spec = deepcopy(spec)
    measurement = spec.get("measurement") or {}
    indicators = measurement.get("indicators") or []
    for indicator in indicators:
        if isinstance(indicator, dict):
            indicator.setdefault("construct_polarity", "positive")
            dtype = indicator.get("measurement_dtype")
            aggregation = indicator.get("aggregation")
            if not isinstance(aggregation, str):
                if dtype == "continuous":
                    indicator["aggregation"] = "mean"
                elif dtype == "count":
                    indicator["aggregation"] = "sum"
                else:
                    indicator["aggregation"] = "last"
                aggregation = indicator["aggregation"]
            if dtype in {"binary", "ordinal", "categorical"} and aggregation not in {
                "first",
                "last",
            }:
                indicator["aggregation"] = "last"
    return spec


def _make_polars_data() -> pl.DataFrame:
    """Long-format polars data for model-spec SSM-validation tests."""
    rng = np.random.default_rng(42)
    n = 30
    anchor_times = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%dT00:00:00Z")
    return pl.DataFrame(
        {
            "indicator": ["mood_score"] * n,
            "value": (rng.standard_normal(n) * 1.5 + 5).tolist(),
            "anchor_time": anchor_times,
            "support_start": anchor_times,
            "support_end": anchor_times,
            "support_kind": ["point"] * n,
            "summary_operator": ["last"] * n,
            "anchor_policy": ["support_end"] * n,
            "observation_window": [None] * n,
        }
    )


@pytest.fixture
def simple_statistical_model_spec() -> dict:
    """Minimal model-spec statistical model spec used by SSM-validation tests."""
    return {
        "likelihoods": [
            {
                "variable": "mood_score",
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Continuous Likert-type scale",
            }
        ],
        "parameters": [
            {
                "name": "rho_mood",
                "role": "ar_coefficient",
                "constraint": "unit_interval",
                "description": "AR(1) coefficient for mood",
            },
            {
                "name": "sigma_mood",
                "role": "residual_sd",
                "constraint": "positive",
                "description": "Residual SD for mood",
            },
        ],
    }


@pytest.fixture
def simple_priors() -> dict:
    """Priors matching ``simple_statistical_model_spec``."""
    return {
        "rho_mood": {
            "parameter": "rho_mood",
            "distribution": "Beta",
            "params": {"alpha": 2.0, "beta": 2.0},
            "sources": [],
            "reasoning": "Weakly informative for AR coefficient",
        },
        "sigma_mood": {
            "parameter": "sigma_mood",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "reasoning": "Weakly informative for residual SD",
        },
    }


@pytest.fixture
def simple_data() -> pd.DataFrame:
    """Tabular fixture aligned with ``simple_statistical_model_spec``."""
    n = 50
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mood_score": rng.normal(5, 1.5, n),
            "mood_score_lag1": rng.normal(5, 1.5, n),
            "subject_id": np.repeat(np.arange(5), 10),
        }
    )
