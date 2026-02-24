"""Shared fixtures for causal SSM tests.

This module provides reusable fixtures to reduce duplication across test files:
- Factory fixtures for creating schema objects (constructs, indicators)
- Stage 1b fixtures (identifiability / proxy resolution)
- Shared SSM data fixtures (lgss_data for recovery tests)

For non-fixture helpers (MockWorkerResult, MockPrediction, make_mock_generate),
see helpers.py.
"""

import jax.numpy as jnp
import jax.random as random
import polars as pl
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
            temporal_scale=granularity,
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
    ) -> Indicator:
        return Indicator(
            name=name,
            construct_name=construct_name,
            how_to_measure=f"Extract {name}",
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
                "temporal_scale": "daily",
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
                "temporal_scale": "daily",
            },
            {
                "name": "Outcome",
                "role": "endogenous",
                "is_outcome": True,
                "description": "The result",
                "temporal_status": "time_varying",
                "temporal_scale": "daily",
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
                "construct_name": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct_name": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
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
                "construct_name": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct_name": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
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
                "construct_name": "Treatment",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "outcome_score",
                "construct_name": "Outcome",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "name": "confounder_proxy",
                "construct_name": "Confounder",
                "how_to_measure": "Proxy measurement for the confounder",
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


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL SPEC FIXTURE (from eval data)
# ══════════════════════════════════════════════════════════════════════════════


def _generate_default_priors(model_spec: dict) -> dict[str, dict]:
    """Generate deterministic default priors for every parameter in a model_spec.

    Creates plausible priors based on the parameter role, suitable for
    smoke-testing the builder pipeline (not for real inference).
    """
    role_defaults: dict[str, tuple[str, dict]] = {
        "ar_coefficient": ("Normal", {"mu": -0.3, "sigma": 0.2}),
        "fixed_effect": ("Normal", {"mu": 0.0, "sigma": 0.5}),
        "residual_sd": ("HalfNormal", {"sigma": 0.5}),
        "loading": ("Normal", {"mu": 0.8, "sigma": 0.3}),
        "correlation": ("Normal", {"mu": 0.0, "sigma": 0.3}),
    }
    fallback = ("Normal", {"mu": 0.0, "sigma": 1.0})

    priors: dict[str, dict] = {}
    for param in model_spec.get("parameters", []):
        name = param["name"]
        role = param.get("role", "")
        dist, params = role_defaults.get(role, fallback)
        priors[name] = {
            "parameter": name,
            "distribution": dist,
            "params": params,
            "reasoning": f"Default test prior for {role} parameter",
            "sources": [],
        }
    return priors


def _generate_synthetic_raw_data(
    model_spec: dict, n_timepoints: int = 50, seed: int = 42
) -> pl.DataFrame:
    """Generate long-format synthetic raw data from a model_spec.

    Produces Gaussian draws per indicator in the format expected by
    ``pivot_to_wide``: columns (indicator, value, timestamp).
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    variables = [lik["variable"] for lik in model_spec.get("likelihoods", [])]

    for t in range(n_timepoints):
        timestamp = f"2024-01-{(t % 28) + 1:02d}T{8 + t % 12:02d}:00:00"
        for var in variables:
            rows.append(
                {
                    "indicator": var,
                    "value": float(rng.normal(0.0, 1.0)),
                    "timestamp": timestamp,
                }
            )

    return pl.DataFrame(rows)


@pytest.fixture(scope="session")
def full_spec_fixture():
    """Complete Stage 4 artifacts from eval data + deterministic spine.

    Loads eval question 1 artifacts, generates default priors and synthetic
    data, injects marginalized correlations, and builds SSMModelBuilder.

    Returns a dict with all intermediate artifacts:
        - model_spec, causal_spec, priors, raw_data
        - builder (fully built SSMModelBuilder)
    """
    import json
    from pathlib import Path

    from causal_ssm_agent.models.ssm_builder import _SUPPORTED_EMISSIONS, build_ssm_builder
    from causal_ssm_agent.orchestrator.schemas_model import ParameterRole
    from causal_ssm_agent.utils.identifiability import inject_marginalized_correlations

    valid_roles = {r.value for r in ParameterRole}
    supported_dists = {d.value for d in _SUPPORTED_EMISSIONS}

    # Load eval artifacts from question 1
    q_dir = Path(__file__).parent.parent / "data" / "eval" / "questions" / "1_resolve-errors-faster"
    with (q_dir / "model_spec.json").open() as f:
        model_spec = json.load(f)
    with (q_dir / "causal_spec.json").open() as f:
        causal_spec = json.load(f)

    # Strip likelihoods with unsupported emission distributions
    model_spec["likelihoods"] = [
        lik
        for lik in model_spec.get("likelihoods", [])
        if lik.get("distribution") in supported_dists
    ]
    # Strip parameters with unsupported roles (e.g. random_intercept_sd)
    model_spec["parameters"] = [
        p for p in model_spec.get("parameters", []) if p.get("role") in valid_roles
    ]
    # Strip random_effects (not yet supported by SSMModelBuilder)
    model_spec.pop("random_effects", None)

    # Inject marginalized correlations
    inject_marginalized_correlations(model_spec, causal_spec)

    # Generate deterministic priors
    priors = _generate_default_priors(model_spec)

    # Generate synthetic raw data
    raw_data = _generate_synthetic_raw_data(model_spec, n_timepoints=50)

    # Build SSMModelBuilder
    builder = build_ssm_builder(
        model_spec=model_spec,
        priors=priors,
        raw_data=raw_data,
        causal_spec=causal_spec,
    )

    return {
        "model_spec": model_spec,
        "causal_spec": causal_spec,
        "priors": priors,
        "raw_data": raw_data,
        "builder": builder,
    }
