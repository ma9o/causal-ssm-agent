"""Shared posterior-predictive test data builders."""

import jax.numpy as jnp
import jax.random as random

from causal_ssm_agent.artifacts import LinkFunction


def make_samples(
    n_draws: int = 20,
    n_latent: int = 2,
    n_manifest: int = 3,
    seed: int = 0,
    drift_diag: float = -0.3,
    diff_sd: float = 0.3,
    obs_sd: float = 0.5,
    with_cint: bool = False,
) -> dict[str, jnp.ndarray]:
    """Build synthetic posterior samples for testing."""
    key = random.PRNGKey(seed)

    k1, *_ = random.split(key, 6)
    drift_base = jnp.eye(n_latent) * drift_diag
    offdiag = random.normal(k1, (n_draws, n_latent, n_latent)) * 0.01
    drift_draws = jnp.broadcast_to(drift_base, (n_draws, n_latent, n_latent)) + offdiag
    diag_idx = jnp.arange(n_latent)
    drift_draws = drift_draws.at[:, diag_idx, diag_idx].set(
        -jnp.abs(drift_draws[:, diag_idx, diag_idx])
    )

    diff_chol = jnp.eye(n_latent) * diff_sd
    diffusion_draws = jnp.broadcast_to(diff_chol, (n_draws, n_latent, n_latent))

    lambda_mat = jnp.zeros((n_manifest, n_latent))
    for i in range(min(n_manifest, n_latent)):
        lambda_mat = lambda_mat.at[i, i].set(1.0)
    for i in range(n_latent, n_manifest):
        lambda_mat = lambda_mat.at[i, 0].set(0.5)

    samples = {
        "drift": drift_draws,
        "diffusion": diffusion_draws,
        "lambda": lambda_mat,
        "manifest_cov": jnp.eye(n_manifest) * obs_sd**2,
        "t0_means": jnp.zeros((n_draws, n_latent)),
        "t0_cov": jnp.eye(n_latent) * 1.0,
    }

    if with_cint:
        samples["cint"] = jnp.zeros((n_draws, n_latent))

    return samples


def complex_mixed_family_config() -> tuple[list[str], list[str], list[int], list[str]]:
    manifest_dists = [
        "gaussian",
        "bernoulli",
        "poisson",
        "student_t",
        "gamma",
        "beta",
        "ordered_logistic",
        "categorical",
        "negative_binomial",
        "gaussian",
    ]
    manifest_links = [
        LinkFunction.IDENTITY.value,
        LinkFunction.LOGIT.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
        LinkFunction.LOG.value,
        LinkFunction.LOGIT.value,
        LinkFunction.CUMULATIVE_LOGIT.value,
        LinkFunction.SOFTMAX.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
    ]
    manifest_level_counts = [0, 0, 0, 0, 0, 0, 4, 4, 0, 0]
    manifest_names = [
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
    ]
    return manifest_dists, manifest_links, manifest_level_counts, manifest_names


def make_complex_mixed_samples(
    *,
    n_draws: int = 12,
    n_latent: int = 4,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    samples = make_samples(
        n_draws=n_draws,
        n_latent=n_latent,
        n_manifest=10,
        seed=seed,
        drift_diag=-0.35,
        diff_sd=0.2,
        obs_sd=0.15,
        with_cint=True,
    )
    samples["lambda"] = jnp.array(
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
    )
    samples["manifest_cov"] = jnp.diag(
        jnp.array([0.12, 0.08, 0.1, 0.18, 0.1, 0.05, 0.08, 0.08, 0.11, 0.12], dtype=jnp.float32)
        ** 2
    )
    samples["t0_cov"] = jnp.eye(n_latent, dtype=jnp.float32) * 0.25
    samples["manifest_means"] = jnp.broadcast_to(
        jnp.array([0.0, -0.3, 0.4, 0.0, -0.2, 0.0, 0.0, 0.1, 0.2, -0.1], dtype=jnp.float32),
        (n_draws, 10),
    )
    samples["obs_df"] = jnp.full((n_draws,), 6.0, dtype=jnp.float32)
    samples["obs_shape"] = jnp.full((n_draws,), 3.0, dtype=jnp.float32)
    samples["obs_r"] = jnp.full((n_draws,), 8.0, dtype=jnp.float32)
    samples["obs_concentration"] = jnp.full((n_draws,), 14.0, dtype=jnp.float32)

    ordered_cutpoints = jnp.zeros((10, 3), dtype=jnp.float32)
    ordered_cutpoints = ordered_cutpoints.at[6].set(jnp.array([-1.2, 0.0, 1.1], dtype=jnp.float32))
    samples["obs_ordered_cutpoints"] = ordered_cutpoints

    cat_intercepts = jnp.zeros((10, 3), dtype=jnp.float32)
    cat_intercepts = cat_intercepts.at[7].set(jnp.array([0.8, -0.1, -0.6], dtype=jnp.float32))
    samples["obs_cat_intercepts"] = cat_intercepts

    cat_slopes = jnp.zeros((10, 3), dtype=jnp.float32)
    cat_slopes = cat_slopes.at[7].set(jnp.array([0.5, -0.2, -0.4], dtype=jnp.float32))
    samples["obs_cat_slopes"] = cat_slopes

    return samples
