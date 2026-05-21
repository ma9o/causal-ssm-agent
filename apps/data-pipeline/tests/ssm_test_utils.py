"""Test helpers for constructing explicit SSMSpec instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np

from nof1_causal_lab.models.ssm import SSMModel, SSMSpec, discretize_system
from nof1_causal_lab.models.ssm.dynamics.composite import linear_drift_spec
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime
from nof1_causal_lab.models.ssm.priors import PriorRegistry, PriorSpec
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
    default_diffusion_block,
    default_input_effect_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.dynamics.composite import CompositeSpec


def make_lgss_data(
    *,
    T: int = 100,
    dt: float = 1.0,
    drift_diag: float = -0.3,
    diff_sd: float = 0.3,
    obs_sd: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Build 1D linear-Gaussian SSM data plus a free-parameter SSMSpec.

    Returns a dict with ``observations``, ``times``, ``spec``, the true
    parameter values, and ``n_latent`` for convenience. Used by recovery
    checks that fit the same canonical 1D model with different inference
    methods.
    """
    n_latent, n_manifest = 1, 1

    true_drift = jnp.array([[drift_diag]])
    true_diff_cov = jnp.array([[diff_sd**2]])
    true_obs_var = jnp.array([[obs_sd**2]])

    Ad, Qd, _ = discretize_system(true_drift, true_diff_cov, None, dt)
    Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
    R_chol = jla.cholesky(true_obs_var, lower=True)

    key = random.PRNGKey(seed)
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
        drift_spec=linear_drift_spec(
            n_latent=n_latent,
            drift_diag_mask=np.ones(n_latent, dtype=bool),
            drift_offdiag_mask=np.zeros((n_latent, n_latent), dtype=bool),
            drift_template=jnp.zeros((n_latent, n_latent)),
            cint_mask=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent),
        ),
        diffusion_block=DiffusionBlockSpec(
            n_latent=n_latent,
            diffusion_chol_mask=np.diag(np.ones(n_latent, dtype=bool)),
            diffusion_chol_template=jnp.eye(n_latent),
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            mask=np.zeros((n_manifest, n_latent), dtype=bool),
            template=jnp.eye(n_manifest, n_latent),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=default_manifest_means_block(n_manifest),
        manifest_chol_block=default_manifest_chol_block(n_manifest),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            mask=np.zeros(n_latent, dtype=bool),
            template=jnp.zeros(n_latent),
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
            diag_mask=np.zeros(n_latent, dtype=bool),
            correlation_mask=np.zeros((n_latent, n_latent), dtype=bool),
            template=jnp.eye(n_latent),
        ),
        input_effect_block=default_input_effect_block(n_latent),
        static_state_sd_block=default_static_state_sd_block(),
    )

    return {
        "observations": observations,
        "times": times,
        "spec": spec,
        "true_drift_diag": drift_diag,
        "true_diff_diag": diff_sd,
        "true_obs_sd": obs_sd,
        "n_latent": n_latent,
    }


def prior_registry(**priors_by_site: PriorSpec) -> PriorRegistry:
    """Build a partial site-keyed prior registry for tests."""
    return PriorRegistry(priors_by_site)


def block_ssm_spec(
    *,
    n_latent: int,
    drift_spec: CompositeSpec,
    n_manifest: int | None = None,
    diffusion_block: DiffusionBlockSpec | None = None,
    lambda_block: SparseMatrixBlockSpec | None = None,
    manifest_means_block: SparseVectorBlockSpec | None = None,
    manifest_chol_block: ManifestCholBlockSpec | None = None,
    t0_means_block: SparseVectorBlockSpec | None = None,
    t0_chol_block: T0CholBlockSpec | None = None,
    input_effect_block: SparseMatrixBlockSpec | None = None,
    static_state_sd_block: SparseVectorBlockSpec | None = None,
    static_factor_loadings: jnp.ndarray | None = None,
    **metadata: Any,
) -> SSMSpec:
    """Build an ``SSMSpec`` from canonical block specs for tests."""
    n_manifest = n_latent if n_manifest is None else n_manifest
    if static_factor_loadings is None:
        static_factor_loadings = jnp.zeros((n_latent, 0), dtype=jnp.float64)
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift_spec=drift_spec,
        diffusion_block=diffusion_block or default_diffusion_block(n_latent),
        lambda_block=lambda_block or default_lambda_block(n_manifest, n_latent),
        manifest_means_block=manifest_means_block or default_manifest_means_block(n_manifest),
        manifest_chol_block=manifest_chol_block or default_manifest_chol_block(n_manifest),
        t0_means_block=t0_means_block or default_t0_means_block(n_latent),
        t0_chol_block=t0_chol_block or default_t0_chol_block(n_latent),
        input_effect_block=input_effect_block or default_input_effect_block(n_latent),
        static_state_sd_block=static_state_sd_block or default_static_state_sd_block(),
        static_factor_loadings=static_factor_loadings,
        **metadata,
    )


def diagonal_diffusion_block(n_latent: int) -> DiffusionBlockSpec:
    """Diagonal-only diffusion: only diagonal entries free, identity template."""
    return DiffusionBlockSpec(
        n_latent=n_latent,
        diffusion_chol_mask=np.diag(np.ones(n_latent, dtype=bool)),
        diffusion_chol_template=jnp.eye(n_latent),
    )


def make_composite_ssm_model(
    drift_spec: CompositeSpec,
    *,
    n_latent: int,
    n_manifest: int,
    H: jnp.ndarray,
    d_meas: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    R: jnp.ndarray,
    manifest_dists: list | None = None,
    manifest_links: list | None = None,
) -> SSMModel:
    """Wrap a ``CompositeSpec`` plus runtime hyperparams into an ``SSMModel``.

    Cholesky-factors the covariance arrays into the templates the
    corresponding block-specs require. The non-drift blocks are fully
    fixed (zero masks) since this helper exists for composite tests
    that condition on these values rather than sample them.
    """
    init_cov = jnp.asarray(init_cov)
    diffusion_cov = jnp.asarray(diffusion_cov)
    R = jnp.asarray(R)

    t0_chol = jnp.linalg.cholesky(init_cov)
    diffusion_chol = jnp.linalg.cholesky(diffusion_cov)
    manifest_chol = jnp.linalg.cholesky(R)

    extra: dict[str, Any] = {}
    if manifest_dists is not None:
        extra["manifest_dists"] = manifest_dists
    if manifest_links is not None:
        extra["manifest_links"] = manifest_links

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift_spec=drift_spec,
        diffusion_block=DiffusionBlockSpec(
            n_latent=n_latent,
            diffusion_chol_mask=np.tri(n_latent, dtype=bool),
            diffusion_chol_template=diffusion_chol,
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            mask=np.zeros((n_manifest, n_latent), dtype=bool),
            template=jnp.asarray(H),
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
            mask=np.zeros(n_manifest, dtype=bool),
            template=jnp.asarray(d_meas),
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
            diag_mask=np.ones(n_manifest, dtype=bool),
            template=manifest_chol,
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            mask=np.ones(n_latent, dtype=bool),
            template=jnp.asarray(init_mean),
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
            diag_mask=np.ones(n_latent, dtype=bool),
            correlation_mask=np.tri(n_latent, k=-1, dtype=bool),
            template=t0_chol,
        ),
        input_effect_block=default_input_effect_block(n_latent),
        static_state_sd_block=default_static_state_sd_block(),
        **extra,
    )
    return SSMModel(spec)


def make_observation_support_runtime(**kwargs: Any) -> ObservationSupportRuntime:
    """Build ObservationSupportRuntime while accepting 2D interval coefficient inputs."""
    support_kinds = kwargs["support_kinds"]
    kwargs.setdefault(
        "summary_operators",
        ["mean" if kind == "interval" else "last" for kind in support_kinds],
    )
    kwargs.setdefault(
        "anchor_policies",
        [
            "support_start" if operator == "first" else "support_end"
            for operator in kwargs["summary_operators"]
        ],
    )
    prev = np.asarray(kwargs["interval_prev_coeffs"], dtype=np.float64)
    curr = np.asarray(kwargs["interval_curr_coeffs"], dtype=np.float64)
    weights = np.asarray(kwargs["interval_weights"], dtype=np.float64)
    if prev.ndim == 2:
        prev = prev[..., None]
        curr = curr[..., None]
        weights = weights[..., None]
    kwargs["interval_prev_coeffs"] = prev
    kwargs["interval_curr_coeffs"] = curr
    kwargs["interval_weights"] = weights
    emission_slots = kwargs.get("emission_slot_indices")
    if emission_slots is None:
        support_end = np.asarray(kwargs["support_end_times"])
        emission_slots = np.where(np.isfinite(support_end), 0, -1).astype(np.int64)
    kwargs["emission_slot_indices"] = emission_slots
    return ObservationSupportRuntime(**kwargs)


def assert_recovery_ci(
    samples: jnp.ndarray,
    true_value: float,
    param_name: str,
    transform=None,
    q_low: float = 5.0,
    q_high: float = 95.0,
) -> None:
    """Assert that a true parameter value falls inside a posterior percentile interval."""
    if transform is not None:
        samples = transform(samples)
    lo = float(jnp.percentile(samples, q_low))
    hi = float(jnp.percentile(samples, q_high))
    assert lo <= true_value <= hi, (
        f"{param_name} {true_value:.2f} outside {q_high - q_low:.0f}% CI [{lo:.3f}, {hi:.3f}]"
    )
