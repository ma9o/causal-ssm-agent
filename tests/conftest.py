"""Shared fixtures for causal SSM tests.

This module provides reusable fixtures to reduce duplication across test files:
- Factory fixtures for creating schema objects (constructs, indicators)
- Stage 1b fixtures (identifiability / proxy resolution)
- Shared SSM data fixtures (lgss_data for recovery tests)

For non-fixture helpers (MockPrediction, make_mock_generate), see helpers.py.
"""

import jax.numpy as jnp
import jax.random as random
import pytest

from causal_ssm_agent.models.ssm import SSMSpec
from causal_ssm_agent.orchestrator.schemas import (
    Construct,
    Indicator,
    Role,
    TemporalStatus,
)

# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def construct_factory():
    """Factory for creating Construct objects.

    Usage:
        def test_something(construct_factory):
            stress = construct_factory("stress", "daily", Role.EXOGENOUS)
            mood = construct_factory("mood", "daily", Role.ENDOGENOUS, is_outcome=True)
    """

    def _make(
        name: str,
        granularity: str | None = "daily",
        role: Role = Role.ENDOGENOUS,
        is_outcome: bool = False,
    ) -> Construct:
        temporal_status = (
            TemporalStatus.TIME_VARYING if granularity else TemporalStatus.TIME_INVARIANT
        )
        return Construct(
            name=name,
            description=f"{name} description",
            role=role,
            is_outcome=is_outcome,
            temporal_status=temporal_status,
            causal_granularity=granularity,
        )

    return _make


@pytest.fixture
def indicator_factory():
    """Factory for creating Indicator objects.

    Usage:
        def test_something(indicator_factory):
            ind = indicator_factory("mood_rating", "mood")
    """

    def _make(
        name: str,
        construct: str,
        granularity: str = "daily",
        dtype: str = "continuous",
        aggregation: str = "mean",
    ) -> Indicator:
        return Indicator(
            name=name,
            construct=construct,
            how_to_measure=f"Extract {name}",
            measurement_granularity=granularity,
            measurement_dtype=dtype,
            aggregation=aggregation,
        )

    return _make


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1B FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def stage1b_simple_latent():
    """Simple chain: Treatment -> Outcome (all observable)."""
    return {
        "constructs": [
            {
                "name": "Treatment",
                "role": "exogenous",
                "description": "The intervention",
                "temporal_status": "time_invariant",
            },
            {
                "name": "Outcome",
                "role": "endogenous",
                "is_outcome": True,
                "description": "The result",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
            },
        ],
        "edges": [
            {
                "cause": "Treatment",
                "effect": "Outcome",
                "description": "Treatment causes Outcome",
            },
        ],
    }


@pytest.fixture
def stage1b_confounded_latent():
    """Confounded: Treatment -> Outcome, Confounder -> Treatment, Confounder -> Outcome."""
    return {
        "constructs": [
            {
                "name": "Treatment",
                "role": "endogenous",
                "description": "The intervention",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
            },
            {
                "name": "Outcome",
                "role": "endogenous",
                "is_outcome": True,
                "description": "The result",
                "temporal_status": "time_varying",
                "causal_granularity": "daily",
            },
            {
                "name": "Confounder",
                "role": "exogenous",
                "description": "Unmeasured common cause",
                "temporal_status": "time_invariant",
            },
        ],
        "edges": [
            {
                "cause": "Treatment",
                "effect": "Outcome",
                "description": "Treatment causes Outcome",
            },
            {
                "cause": "Confounder",
                "effect": "Treatment",
                "description": "Confounder affects Treatment",
            },
            {
                "cause": "Confounder",
                "effect": "Outcome",
                "description": "Confounder affects Outcome",
            },
        ],
    }


@pytest.fixture
def stage1b_measurement_all_observed():
    """Measurement model with indicators for Treatment and Outcome."""
    return {
        "indicators": [
            {
                "name": "treatment_dose",
                "construct": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


@pytest.fixture
def stage1b_measurement_missing_confounder():
    """Measurement model missing the confounder (non-identifiable)."""
    return {
        "indicators": [
            {
                "name": "treatment_dose",
                "construct": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


@pytest.fixture
def stage1b_measurement_with_confounder():
    """Measurement model with confounder indicator (identifiable)."""
    return {
        "indicators": [
            {
                "name": "treatment_dose",
                "construct": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "confounder_proxy",
                "construct": "Confounder",
                "how_to_measure": "Proxy measurement for the confounder",
                "measurement_granularity": "daily",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]
    }


@pytest.fixture
def stage1b_proxy_response_success():
    """Successful proxy response that adds confounder indicator."""
    return {
        "new_proxies": [
            {
                "construct": "Confounder",
                "indicators": ["confounder_proxy"],
                "justification": "This proxy captures the confounder",
            }
        ],
        "unfeasible_confounders": [],
    }


@pytest.fixture
def stage1b_proxy_response_empty():
    """Empty proxy response (no proxies found)."""
    return {
        "new_proxies": [],
        "unfeasible_confounders": ["Confounder"],
    }


@pytest.fixture
def stage1b_dummy_chunks():
    """Dummy data chunks for measurement model proposal."""
    return [
        "Day 1: Patient took 10mg treatment, outcome score was 5.",
        "Day 2: Patient took 15mg treatment, outcome score was 7.",
        "Day 3: Patient took 10mg treatment, outcome score was 6.",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SSM DATA FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def lgss_data():
    """1D Linear Gaussian SSM data for smoke and recovery tests.

    Generates T=100 observations from a 1D LGSS with:
    - drift = -0.3 (stable AR)
    - diffusion SD = 0.3
    - observation SD = 0.5

    Used by TestHessMC2Smoke, TestPGASSmoke, TestTemperedSMCSmoke.
    """
    import jax.scipy.linalg as jla

    from causal_ssm_agent.models.ssm import discretize_system

    n_latent, n_manifest = 1, 1
    T, dt = 100, 1.0

    true_drift = jnp.array([[-0.3]])  # stable AR
    true_diff_cov = jnp.array([[0.3**2]])  # process noise var
    true_obs_var = jnp.array([[0.5**2]])  # observation noise var

    Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
    Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
    R_chol = jla.cholesky(true_obs_var, lower=True)

    key = random.PRNGKey(42)
    states = [jnp.zeros(n_latent)]
    for _ in range(T - 1):
        key, nk = random.split(key)
        states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
    latent = jnp.stack(states)

    key, obs_key = random.split(key)
    observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
    times = jnp.arange(T, dtype=float) * dt

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        lambda_mat=jnp.eye(n_manifest, n_latent),
        manifest_means=jnp.zeros(n_manifest),
        diffusion="diag",
        t0_means=jnp.zeros(n_latent),
        t0_var=jnp.eye(n_latent),
    )

    return {
        "observations": observations,
        "times": times,
        "spec": spec,
        "true_drift_diag": -0.3,
        "true_diff_diag": 0.3,
        "true_obs_sd": 0.5,
        "n_latent": n_latent,
    }
