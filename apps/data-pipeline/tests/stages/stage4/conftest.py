"""Stage 4 shared fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_model_spec() -> dict:
    """A minimal model spec for testing."""
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
    """Simple priors matching the model spec."""
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
    """Simple test data with lagged columns."""
    n = 50
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mood_score": rng.normal(5, 1.5, n),
            "mood_score_lag1": rng.normal(5, 1.5, n),
            "subject_id": np.repeat(np.arange(5), 10),
        }
    )
