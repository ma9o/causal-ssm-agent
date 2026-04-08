"""Shared fixtures for causal SSM tests.

This module provides reusable fixtures to reduce duplication across test files:
- Factory fixtures for creating schema objects (constructs, indicators)
- Stage 1b fixtures (identifiability / proxy resolution)
- Shared SSM data fixtures (lgss_data for recovery tests)

For non-fixture helpers (make_mock_generate, assert_recovery_ci),
see helpers.py.
"""

import jax.numpy as jnp
import jax.random as random
import pytest

from causal_ssm_agent.models.ssm import SSMSpec, full_drift_mask, zero_loading_mask
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
            stress = construct_factory("stress", Role.EXOGENOUS)
            mood = construct_factory("mood", Role.ENDOGENOUS, is_outcome=True)
    """

    def _make(
        name: str,
        role: Role = Role.ENDOGENOUS,
        is_outcome: bool = False,
        temporal_status: TemporalStatus = TemporalStatus.TIME_VARYING,
    ) -> Construct:
        return Construct(
            name=name,
            description=f"{name} description",
            role=role,
            is_outcome=is_outcome,
            temporal_status=temporal_status,
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
        construct_name: str,
        dtype: str = "continuous",
        aggregation: str = "mean",
        construct_polarity: str = "positive",
        ordinal_levels: list[str] | None = None,
        source_columns: list[str] | None = None,
        extraction_mode: str = "semantic",
    ) -> Indicator:
        # Auto-provide ordinal_levels for ordinal dtype if not specified
        if dtype == "ordinal" and ordinal_levels is None:
            ordinal_levels = ["low", "medium", "high"]
        return Indicator(
            name=name,
            construct_name=construct_name,
            construct_polarity=construct_polarity,
            how_to_measure=f"Extract {name}",
            measurement_dtype=dtype,
            aggregation=aggregation,
            ordinal_levels=ordinal_levels,
            source_columns=source_columns or [name],
            extraction_mode=extraction_mode,
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
            },
            {
                "name": "Outcome",
                "role": "endogenous",
                "is_outcome": True,
                "description": "The result",
                "temporal_status": "time_varying",
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
        "model_clock": "1d",
        "indicators": [
            {
                "name": "treatment_dose",
                "construct_name": "Treatment",
                "construct_polarity": "positive",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["treatment_dose"],
            },
            {
                "name": "outcome_score",
                "construct_name": "Outcome",
                "construct_polarity": "positive",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["outcome_score"],
            },
        ],
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
        drift_mask=full_drift_mask(n_latent),
        lambda_mask=zero_loading_mask(n_manifest, n_latent),
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
