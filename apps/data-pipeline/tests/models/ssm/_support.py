"""Shared SSM test builders, simulators, and reference models.

Consolidates what used to live in ``autoreparam_support.py``,
``prior_predictive_support.py``, and ``posterior_predictive_support.py``.
Imported from both fast (``tests/models/ssm/``) and slow
(``tests/slow/models/ssm/``) SSM test suites.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist

from nof1_causal_lab.artifacts import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm import SSMSpec
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from tests.ssm_test_utils import (
    default_input_effect_block,
    default_static_state_sd_block,
    dense_matrix_dynamics_spec,
)

# ══════════════════════════════════════════════════════════════════════════════
# AUTOREPARAM
# ══════════════════════════════════════════════════════════════════════════════


def simple_normal_model():
    x = numpyro.sample("x", dist.Normal(0.0, 1.0))
    y = numpyro.sample("y", dist.Normal(x, 0.5))
    numpyro.sample("obs", dist.Normal(y, 0.1), obs=jnp.array(1.0))


# ══════════════════════════════════════════════════════════════════════════════
# POSTERIOR-PREDICTIVE SAMPLES
# ══════════════════════════════════════════════════════════════════════════════


def make_samples(
    n_draws: int = 20,
    n_latent: int = 2,
    n_manifest: int = 3,
    seed: int = 0,
    decay_diag: float = -0.3,
    diff_sd: float = 0.3,
    obs_sd: float = 0.5,
    with_cint: bool = False,
) -> dict[str, jnp.ndarray]:
    """Build synthetic posterior samples for testing."""
    key = random.PRNGKey(seed)

    k1, *_ = random.split(key, 6)
    dynamics_base = jnp.eye(n_latent) * decay_diag
    offdiag = random.normal(k1, (n_draws, n_latent, n_latent)) * 0.01
    dynamics_draws = jnp.broadcast_to(dynamics_base, (n_draws, n_latent, n_latent)) + offdiag
    diag_idx = jnp.arange(n_latent)
    dynamics_draws = dynamics_draws.at[:, diag_idx, diag_idx].set(
        -jnp.abs(dynamics_draws[:, diag_idx, diag_idx])
    )

    diff_chol = jnp.eye(n_latent) * diff_sd
    diffusion_draws = jnp.broadcast_to(diff_chol, (n_draws, n_latent, n_latent))

    lambda_mat = jnp.zeros((n_manifest, n_latent))
    for i in range(min(n_manifest, n_latent)):
        lambda_mat = lambda_mat.at[i, i].set(1.0)
    for i in range(n_latent, n_manifest):
        lambda_mat = lambda_mat.at[i, 0].set(0.5)

    samples = {
        "drift": dynamics_draws,
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
        decay_diag=-0.35,
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


# ══════════════════════════════════════════════════════════════════════════════
# PRIOR-PREDICTIVE RUNTIME SPEC
# ══════════════════════════════════════════════════════════════════════════════


def complex_mixed_runtime_spec() -> SSMSpec:
    n_latent = 4
    n_manifest = 10
    coupling_template = jnp.array(
        [
            [-0.45, 0.0, 0.0, 0.0],
            [0.08, -0.35, 0.0, 0.0],
            [0.02, 0.06, -0.4, 0.0],
            [0.0, 0.03, 0.05, -0.3],
        ],
        dtype=jnp.float32,
    )
    lambda_template = jnp.array(
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
    manifest_means_template = jnp.array(
        [0.0, -0.3, 0.4, 0.0, -0.2, 0.0, 0.0, 0.1, 0.2, -0.1],
        dtype=jnp.float32,
    )
    manifest_chol_template = jnp.diag(
        jnp.array(
            [0.12, 0.08, 0.1, 0.18, 0.1, 0.05, 0.08, 0.08, 0.11, 0.12],
            dtype=jnp.float32,
        )
        ** 2
    )
    t0_chol_template = jnp.eye(n_latent, dtype=jnp.float32) * 0.25
    diffusion_template = jnp.diag(jnp.array([0.2, 0.18, 0.16, 0.14], dtype=jnp.float32))
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dense_matrix_dynamics_spec(
            n_latent=n_latent,
            decay_support=np.zeros(n_latent, dtype=bool),
            edge_support=np.zeros((n_latent, n_latent), dtype=bool),
            coupling_template=coupling_template,
            intercept_support=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent, dtype=jnp.float32),
        ),
        diffusion_block=DiffusionBlockSpec(
            n_latent=n_latent,
            diffusion_chol_support=np.tri(n_latent, dtype=bool),
            diffusion_chol_template=diffusion_template,
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            free_support=np.zeros((n_manifest, n_latent), dtype=bool),
            template=lambda_template,
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=n_manifest,
            free_support=np.zeros(n_manifest, dtype=bool),
            template=manifest_means_template,
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=n_manifest,
            diag_support=np.ones(n_manifest, dtype=bool),
            template=manifest_chol_template,
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            free_support=np.ones(n_latent, dtype=bool),
            template=jnp.zeros(n_latent, dtype=jnp.float32),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=n_latent,
            diag_support=np.ones(n_latent, dtype=bool),
            correlation_support=np.tri(n_latent, k=-1, dtype=bool),
            template=t0_chol_template,
        ),
        input_effect_block=default_input_effect_block(n_latent),
        static_state_sd_block=default_static_state_sd_block(),
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
