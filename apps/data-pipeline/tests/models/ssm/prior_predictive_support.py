"""Shared prior-predictive test spec builders."""

import jax.numpy as jnp

from causal_ssm_agent.artifacts import LinkFunction
from causal_ssm_agent.distributions import DistributionFamily
from causal_ssm_agent.models.ssm.model import SSMSpec
from tests.ssm_test_utils import make_ssm_spec


def complex_mixed_runtime_spec() -> SSMSpec:
    return make_ssm_spec(
        n_latent=4,
        n_manifest=10,
        drift=jnp.array(
            [
                [-0.45, 0.0, 0.0, 0.0],
                [0.08, -0.35, 0.0, 0.0],
                [0.02, 0.06, -0.4, 0.0],
                [0.0, 0.03, 0.05, -0.3],
            ],
            dtype=jnp.float32,
        ),
        diffusion=jnp.diag(jnp.array([0.2, 0.18, 0.16, 0.14], dtype=jnp.float32)),
        cint=jnp.zeros(4, dtype=jnp.float32),
        lambda_mat=jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.3, 0.4, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0],
                [0.2, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.0],
                [0.0, 0.0, 0.5, 0.3],
                [0.0, 0.0, 0.7, 0.0],
                [0.0, 0.2, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.9],
                [0.1, 0.0, 0.2, 0.8],
            ],
            dtype=jnp.float32,
        ),
        manifest_means=jnp.array(
            [0.0, -0.3, 0.4, 0.0, -0.2, 0.0, 0.0, 0.1, 0.2, -0.1],
            dtype=jnp.float32,
        ),
        manifest_var=jnp.diag(
            jnp.array([0.12, 0.08, 0.1, 0.18, 0.1, 0.05, 0.08, 0.08, 0.11, 0.12], dtype=jnp.float32)
            ** 2
        ),
        t0_means=jnp.zeros(4, dtype=jnp.float32),
        t0_var=jnp.eye(4, dtype=jnp.float32) * 0.25,
        manifest_dists=[
            DistributionFamily.GAUSSIAN,
            DistributionFamily.BERNOULLI,
            DistributionFamily.POISSON,
            DistributionFamily.STUDENT_T,
            DistributionFamily.GAMMA,
            DistributionFamily.BETA,
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.GAUSSIAN,
        ],
        manifest_links=[
            LinkFunction.IDENTITY,
            LinkFunction.LOGIT,
            LinkFunction.LOG,
            LinkFunction.IDENTITY,
            LinkFunction.LOG,
            LinkFunction.LOGIT,
            LinkFunction.CUMULATIVE_LOGIT,
            LinkFunction.SOFTMAX,
            LinkFunction.LOG,
            LinkFunction.IDENTITY,
        ],
        manifest_level_counts=[0, 0, 0, 0, 0, 0, 4, 4, 0, 0],
        latent_names=["stress", "adherence", "sleep", "focus"],
        manifest_names=[
            "stress_cont",
            "adherence_flag",
            "steps_count",
            "fatigue_t",
            "screen_gap",
            "sleep_efficiency",
            "symptom_severity",
            "coping_style",
            "rumination_count",
            "focus_cont",
        ],
    )
