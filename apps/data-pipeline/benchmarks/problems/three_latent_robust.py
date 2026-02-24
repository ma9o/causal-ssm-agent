"""Three-latent robust recovery problem: Arousal -> Valence -> Engagement.

Ground truth: 3 latent processes with lower-triangular drift, diagonal diffusion,
5 Student-t indicators (3 identity + 2 cross-loading), and free continuous intercepts.

The Student-t observation noise (df=5) makes this a non-Gaussian model that
exercises the particle filter likelihood path. Heavy tails test robustness
of inference methods to outliers.
"""

from __future__ import annotations

import jax.numpy as jnp

from causal_ssm_agent.models.ssm import SSMPriors, SSMSpec
from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

from .four_latent import RecoveryProblem


def _make_three_latent_robust() -> RecoveryProblem:
    n_latent, n_manifest, dt = 3, 5, 0.5
    names = ["Arousal", "Valence", "Engage"]

    true_drift = jnp.array(
        [
            [-0.6, 0.0, 0.0],
            [0.25, -0.4, 0.0],
            [0.0, 0.3, -0.5],
        ]
    )
    true_diff_diag = jnp.array([0.35, 0.25, 0.3])
    true_cint = jnp.array([0.4, 0.0, 0.2])

    # 5 manifests: 3 identity + 2 cross-loading rows
    true_lambda = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.6, 0.0],
            [0.0, 0.4, 0.7],
        ]
    )
    true_manifest_means = jnp.array([0.0, 0.0, 0.0, 1.0, 1.5])
    true_mvar_diag = jnp.array([0.2, 0.15, 0.15, 0.2, 0.2])
    true_obs_df = 5.0

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift="free",
        diffusion="diag",
        cint="free",
        lambda_mat="free",
        manifest_means="free",
        manifest_var="diag",
        manifest_dist=DistributionFamily.STUDENT_T,
        t0_means="free",
        t0_var="diag",
        latent_names=names,
    )
    priors = SSMPriors(
        drift_diag={"mu": -0.5, "sigma": 0.5},
        drift_offdiag={"mu": 0.0, "sigma": 0.5},
        diffusion_diag={"sigma": 0.5},
        cint={"mu": 0.0, "sigma": 1.0},
        lambda_free={"mu": 0.5, "sigma": 0.5},
        manifest_means={"mu": 0.0, "sigma": 2.0},
        manifest_var_diag={"sigma": 0.5},
    )

    return RecoveryProblem(
        true_drift=true_drift,
        true_diff_diag=true_diff_diag,
        true_cint=true_cint,
        true_lambda=true_lambda,
        true_manifest_means=true_manifest_means,
        true_mvar_diag=true_mvar_diag,
        latent_names=names,
        n_latent=n_latent,
        n_manifest=n_manifest,
        dt=dt,
        spec=spec,
        priors=priors,
        manifest_dist="student_t",
        extra_true_params={"obs_df": true_obs_df},
    )


THREE_LATENT_ROBUST = _make_three_latent_robust()
