"""Shared fixtures + re-exports for the surviving model-spec tests.

Consumed by ``conftest.py`` (the ``simple_*`` fixtures) and
``test_ssm_validation.py`` (pure builders + third-party re-exports).
"""

# ruff: noqa: F401 — this module deliberately re-exports names for its consumers.

from copy import deepcopy
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nof1_causal_lab.artifacts.prior import PriorValidationResult
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_priors as compile_ssm_priors,
)
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_ssm_inputs_from_statistical_model_spec,
)


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
def simple_statistical_model_spec() -> dict[str, Any]:
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
def simple_priors() -> dict[str, Any]:
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
