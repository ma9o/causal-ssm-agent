"""Kernel layer: pre-resolved callables for SSM inference.

Separates the specification domain (SSMSpec: serializable enums for web UI)
from the inference domain (kernels: bound JAX callables). Kernels are built
once per likelihood evaluation from spec enums + sampled hyperparameters,
then passed to all backend internals.

ObservationKernel: p(y_t | x_t) — emission log-prob, inverse link, EKF variance.
TransitionKernel: p(x_t | x_{t-1}) — process noise sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import jax.scipy.stats as jstats
import numpy as np

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter

from .emissions import (
    build_composite_mean_log_prob_fn,
    get_mean_param_log_prob_fn,
)
from .observation_dispatch import (
    get_emission_fn,
    get_emission_score_weight_fn,
)

logger = get_prefect_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
        ObservationOperator,
    )
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

# =============================================================================
# Kernel dataclasses
# =============================================================================


@dataclass(frozen=True)
class ObservationKernel:
    """Pre-resolved observation model for SSM inference.

    Built once from DistributionFamily + LinkFunction + sampled hyperparameters.
    Consumed by the marginal likelihood and blocked MCMC backends.

    Attributes:
        emission_fn: Log-probability (y, z, H, d, R, mask) -> scalar.
        response_fn: Inverse link, maps linear predictor to mean (elementwise).
        variance_fn: Maps predicted mean to (n_m, n_m) pseudo-covariance for
            EKF linearization. Diagonal for GLM families; full manifest_cov
            for Gaussian/Student-t.
        is_gaussian: Whether the observation family is Gaussian.
    """

    emission_fn: Callable
    response_fn: Callable
    variance_fn: Callable
    is_gaussian: bool
    emission_grad_hess_fn: Callable  # (y, z, H, d, R, mask) → (g_z: (D,), neg_H_z: (D,D))


@dataclass(frozen=True)
class TransitionKernel:
    """Pre-resolved process noise model for SSM transition dynamics.

    Callers compute the deterministic mean (Ad @ x + cd) and Cholesky of Qd,
    then call sample_noise_fn to generate the stochastic perturbation.

    Attributes:
        sample_noise_fn: (key, chol_Q) -> (n,) noise vector (Cholesky applied).
        is_gaussian: Whether dynamics are Gaussian.
    """

    sample_noise_fn: Callable
    is_gaussian: bool


@dataclass(frozen=True)
class TransitionSemantics:
    """Compiled process-noise semantics for one latent block."""

    per_var_dists: tuple[DistributionFamily, ...]
    dispatch_mode: str
    gaussian_idx: tuple[int, ...] = ()
    sampled_idx: tuple[int, ...] = ()
    sampled_block_dist: DistributionFamily | None = None

    @property
    def is_gaussian(self) -> bool:
        return self.dispatch_mode == DistributionFamily.GAUSSIAN.value

    @property
    def is_mixed(self) -> bool:
        return self.dispatch_mode == "mixed"


@dataclass(frozen=True)
class MeasurementSemantics:
    """Compiled measurement semantics shared across inference backends."""

    obs_kernel: ObservationKernel
    mean_log_prob_fn: Callable | None
    observation_operator: ObservationOperator
    manifest_dists: tuple[DistributionFamily, ...]
    manifest_links: tuple[LinkFunction, ...]

    @property
    def requires_interval_summary_handling(self) -> bool:
        return self.observation_operator.requires_interval_summary_handling


# =============================================================================
# Response functions (inverse links)
# =============================================================================


def _response_identity(eta: jnp.ndarray) -> jnp.ndarray:
    return eta


def _response_exp(eta: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(eta)


def _response_sigmoid(eta: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(eta)


def _response_probit(eta: jnp.ndarray) -> jnp.ndarray:
    return jstats.norm.cdf(eta)


def _response_inverse(eta: jnp.ndarray) -> jnp.ndarray:
    valid_eta = jnp.isfinite(eta) & (eta > 0.0)
    safe_eta = jnp.where(valid_eta, eta, 1.0)
    response = 1.0 / safe_eta
    return jnp.where(valid_eta, response, jnp.nan)


_RESPONSE_FNS: dict[LinkFunction, Callable] = {
    LinkFunction.IDENTITY: _response_identity,
    LinkFunction.LOG: _response_exp,
    LinkFunction.LOGIT: _response_sigmoid,
    LinkFunction.PROBIT: _response_probit,
    LinkFunction.INVERSE: _response_inverse,
}


def _slice_observation_extra_params(
    extra_params: dict | None, ch_indices: list[int]
) -> dict | None:
    if extra_params is None:
        return None

    sliced: dict = {}
    idx = jnp.array(ch_indices, dtype=jnp.int64)
    for key, value in extra_params.items():
        if (
            hasattr(value, "ndim")
            and hasattr(value, "shape")
            and value.ndim >= 1
            and value.shape[0] == len(idx)
        ):
            sliced[key] = value
            continue
        if hasattr(value, "ndim") and hasattr(value, "shape") and value.ndim >= 1:
            try:
                if value.shape[0] >= len(ch_indices):
                    sliced[key] = value[idx]
                    continue
            except TypeError:
                logger.warning("Unexpected type for kernel parameter %r during slicing", key)
        sliced[key] = value
    return sliced


# =============================================================================
# Emission gradient/Hessian factories for IEKS (analytical, GPU-compatible)
# =============================================================================


def _make_glm_grad_hess(score_weight_fn: Callable) -> Callable:
    """Build emission_grad_hess_fn for GLM families (diagonal Hessian in η-space).

    For dist with element-wise log p(y_j | η_j), the chain rule gives:
        g_z = H^T g_eta,   neg_H_z = H^T diag(w_eta) H
    which is always PSD when w_eta >= 0.
    """

    def emission_grad_hess_fn(y_t, z_t, H, d, _R, mask_t):
        eta = H @ z_t + d
        g_eta, w_eta = score_weight_fn(y_t, eta, mask_t)
        g_z = H.T @ g_eta
        neg_H_z = H.T @ (w_eta[:, None] * H)
        return g_z, symmetrize(neg_H_z)

    return emission_grad_hess_fn


def _make_student_t_grad_hess(df: float) -> Callable:
    """Build emission_grad_hess_fn for Student-t (scale extracted from diag(R))."""

    def emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t):
        eta = H @ z_t + d
        scale_diag = jnp.sqrt(jnp.diag(R))
        residual = y_t - eta
        sig2 = scale_diag**2
        denom = df * sig2 + residual**2
        g_eta = (df + 1.0) * residual / denom * mask_t
        w_eta = jnp.maximum((df + 1.0) * (df * sig2 - residual**2) / (denom**2), 0.0) * mask_t
        g_z = H.T @ g_eta
        neg_H_z = H.T @ (w_eta[:, None] * H)
        return g_z, symmetrize(neg_H_z)

    return emission_grad_hess_fn


def _make_gaussian_grad_hess() -> Callable:
    """Build emission_grad_hess_fn for Gaussian (full R, exact analytical form)."""
    from nof1_causal_lab.models.ssm.inference.targets.base import (
        MISSING_DATA_LARGE_VAR,
    )

    def emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t):
        eta = H @ z_t + d
        residual = (y_t - eta) * mask_t
        R_adj = R + jnp.diag((1.0 - mask_t) * MISSING_DATA_LARGE_VAR)
        R_adj = symmetrize_with_jitter(R_adj)
        g_z = H.T @ jla.solve(R_adj, residual, assume_a="pos")
        neg_H_z = H.T @ jla.solve(R_adj, H, assume_a="pos")
        return g_z, symmetrize(neg_H_z)

    return emission_grad_hess_fn


# =============================================================================
# ObservationKernel factory
# =============================================================================


def build_observation_kernel(
    dist: DistributionFamily,
    link: LinkFunction,
    extra_params: dict | None = None,
    manifest_cov: jnp.ndarray | None = None,
) -> ObservationKernel:
    """Build an ObservationKernel from spec enums + sampled hyperparameters.

    This is the single resolution point: enums and runtime parameters go in,
    pre-bound callables come out. Called once per likelihood evaluation.

    Args:
        dist: Distribution family enum.
        link: Link function enum.
        extra_params: Sampled hyperparameters (obs_df, obs_shape, obs_r,
            obs_concentration, etc.).
        manifest_cov: Measurement noise covariance matrix. Required for
            Gaussian/Student-t (used as EKF pseudo-covariance). Ignored
            for GLM families.
    """
    from nof1_causal_lab.models.ssm.inference.targets.observation_families import FAMILY_REGISTRY

    extra_params = extra_params or {}
    family_spec = FAMILY_REGISTRY.get(dist)
    if family_spec is None:
        raise ValueError(
            f"No observation kernel for dist={dist!r}. "
            f"Supported: {sorted(d.value for d in FAMILY_REGISTRY)}"
        )

    # Emission log-prob (delegates to existing canonical functions)
    emission_fn = get_emission_fn(dist, extra_params, link=link)

    # Response function (inverse link)
    if family_spec.make_response_fn is not None:
        response_fn = family_spec.make_response_fn(extra_params)
    else:
        response_fn = _RESPONSE_FNS.get(link)
        if response_fn is None:
            raise ValueError(
                f"No response function for link={link!r}. Supported: {list(_RESPONSE_FNS.keys())}"
            )

    # Variance function + is_gaussian flag
    is_gaussian = dist == DistributionFamily.GAUSSIAN
    variance_fn = family_spec.make_variance_fn(extra_params, manifest_cov)

    # Build emission_grad_hess_fn (analytical, avoids jax.hessian on GPU)
    if family_spec.grad_hess_strategy == "gaussian":
        emission_grad_hess_fn = _make_gaussian_grad_hess()
    elif family_spec.grad_hess_strategy == "student_t":
        emission_grad_hess_fn = _make_student_t_grad_hess(extra_params.get("obs_df", 5.0))
    else:  # "glm"
        sw_fn = get_emission_score_weight_fn(dist, extra_params, link=link)
        assert sw_fn is not None, f"No analytical score/weight fn for dist={dist!r}"
        emission_grad_hess_fn = _make_glm_grad_hess(sw_fn)

    return ObservationKernel(
        emission_fn=emission_fn,
        response_fn=response_fn,
        variance_fn=variance_fn,
        is_gaussian=is_gaussian,
        emission_grad_hess_fn=emission_grad_hess_fn,
    )


# =============================================================================
# TransitionKernel factory
# =============================================================================


def _make_gaussian_noise() -> Callable:
    def sample_noise(key: jax.Array, chol_Q: jnp.ndarray) -> jnp.ndarray:
        n = chol_Q.shape[0]
        return chol_Q @ random.normal(key, (n,))

    return sample_noise


def _make_student_t_noise(df: float) -> Callable:
    def sample_noise(key: jax.Array, chol_Q: jnp.ndarray) -> jnp.ndarray:
        n = chol_Q.shape[0]
        df_safe = jnp.maximum(df, 2.1)
        key_z, key_chi2 = random.split(key)
        z = random.normal(key_z, (n,))
        chi2 = jnp.maximum(random.gamma(key_chi2, df_safe / 2.0) * 2.0, 1e-8)
        scale = jnp.sqrt((df_safe - 2.0) / chi2)
        return chol_Q @ (z * scale)

    return sample_noise


def _make_mixed_noise(
    per_var_dists: tuple[DistributionFamily, ...],
    df: float,
) -> Callable:
    """Build a mixed per-coordinate noise sampler with unit-variance standardized shocks.

    Each latent innovation coordinate draws a standardized shock according to
    its declared family, then the full innovation covariance is induced by
    left-multiplying with ``chol_Q``.
    """

    supported = {DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T}
    unsupported = sorted({dist.value for dist in per_var_dists if dist not in supported})
    if unsupported:
        raise ValueError(
            "Mixed transition kernels only support gaussian/student_t families. "
            f"Received: {unsupported}"
        )

    student_t_mask = jnp.asarray(
        [dist == DistributionFamily.STUDENT_T for dist in per_var_dists],
        dtype=bool,
    )
    has_student_t = any(student_t_mask.tolist())

    def sample_noise(key: jax.Array, chol_Q: jnp.ndarray) -> jnp.ndarray:
        n = chol_Q.shape[0]
        key_z, key_chi2 = random.split(key)
        z = random.normal(key_z, (n,))
        if not has_student_t:
            standardized = z
        else:
            df_safe = jnp.maximum(df, 2.1)
            chi2 = jnp.maximum(
                random.gamma(key_chi2, jnp.full((n,), df_safe / 2.0, dtype=chol_Q.dtype)) * 2.0,
                1e-8,
            )
            scale = jnp.sqrt((df_safe - 2.0) / chi2)
            standardized = jnp.where(student_t_mask, z * scale, z)
        return chol_Q @ standardized

    return sample_noise


def compile_transition_semantics(
    diffusion_dists: TransitionSemantics
    | list[DistributionFamily | str]
    | tuple[DistributionFamily | str, ...],
    n_latent: int | None = None,
) -> TransitionSemantics:
    """Resolve per-latent diffusion families into a compiled description."""
    if isinstance(diffusion_dists, TransitionSemantics):
        semantics = diffusion_dists
    else:
        if not isinstance(diffusion_dists, (list, tuple)):
            raise TypeError(
                "compile_transition_semantics requires a per-latent diffusion_dists list "
                "or a compiled TransitionSemantics."
            )
        per_var_dists = tuple(
            dist if isinstance(dist, DistributionFamily) else DistributionFamily(dist)
            for dist in diffusion_dists
        )
        if n_latent is not None and len(per_var_dists) != n_latent:
            raise ValueError(
                f"diffusion_dists length {len(per_var_dists)} does not match n_latent={n_latent}"
            )

        unique = set(per_var_dists)
        gaussian_idx = tuple(
            i for i, family in enumerate(per_var_dists) if family == DistributionFamily.GAUSSIAN
        )
        sampled_idx = tuple(
            i for i, family in enumerate(per_var_dists) if family != DistributionFamily.GAUSSIAN
        )
        sampled_unique = {per_var_dists[i] for i in sampled_idx}

        if unique == {DistributionFamily.GAUSSIAN}:
            dispatch_mode = DistributionFamily.GAUSSIAN.value
        elif len(unique) == 1:
            dispatch_mode = next(iter(unique)).value
        else:
            dispatch_mode = "mixed"

        semantics = TransitionSemantics(
            per_var_dists=per_var_dists,
            dispatch_mode=dispatch_mode,
            gaussian_idx=gaussian_idx,
            sampled_idx=sampled_idx,
            sampled_block_dist=next(iter(sampled_unique)) if len(sampled_unique) == 1 else None,
        )

    if n_latent is not None and len(semantics.per_var_dists) != n_latent:
        raise ValueError(
            f"transition semantics width {len(semantics.per_var_dists)} does not match n_latent={n_latent}"
        )
    return semantics


def build_transition_kernel(
    diffusion_dists: TransitionSemantics
    | list[DistributionFamily | str]
    | tuple[DistributionFamily | str, ...],
    extra_params: dict | None = None,
) -> TransitionKernel:
    """Build a TransitionKernel from spec enum + sampled hyperparameters.

    Args:
        diffusion_dists: Compiled semantics or canonical per-latent diffusion families.
        extra_params: Sampled hyperparameters (proc_df, etc.).
    """
    extra_params = extra_params or {}
    semantics = compile_transition_semantics(diffusion_dists)

    if semantics.dispatch_mode == DistributionFamily.GAUSSIAN.value:
        return TransitionKernel(
            sample_noise_fn=_make_gaussian_noise(),
            is_gaussian=True,
        )
    if semantics.dispatch_mode == DistributionFamily.STUDENT_T.value:
        df = extra_params.get("proc_df", 5.0)
        return TransitionKernel(
            sample_noise_fn=_make_student_t_noise(df),
            is_gaussian=False,
        )
    if semantics.dispatch_mode == "mixed":
        df = extra_params.get("proc_df", 5.0)
        return TransitionKernel(
            sample_noise_fn=_make_mixed_noise(semantics.per_var_dists, df),
            is_gaussian=False,
        )
    raise ValueError(
        "No transition kernel for transition dispatch mode="
        f"{semantics.dispatch_mode!r}. Supported: gaussian, student_t, mixed gaussian/student_t."
    )


# =============================================================================
# Composite ObservationKernel for per-channel heterogeneous distributions
# =============================================================================


def build_composite_observation_kernel(
    dists: list[DistributionFamily],
    links: list[LinkFunction],
    extra_params: dict | None = None,
    manifest_cov: jnp.ndarray | None = None,
) -> ObservationKernel:
    """Build an ObservationKernel that handles per-channel heterogeneous distributions.

    Groups channels by unique (dist, link) combination, builds one kernel per group,
    and composes their emission_fn and emission_grad_hess_fn by dispatching per group.

    When all channels share the same (dist, link), delegates to the standard
    build_observation_kernel for zero overhead.

    Args:
        dists: Per-channel distribution families (length n_manifest).
        links: Per-channel link functions (length n_manifest).
        extra_params: Sampled hyperparameters (obs_df, obs_shape, etc.).
        manifest_cov: Measurement noise covariance matrix for Gaussian / Student-t
            subgroups inside a heterogeneous manifest family layout.
    """
    n_manifest = len(dists)
    if n_manifest != len(links):
        raise ValueError(f"dists ({len(dists)}) and links ({len(links)}) must have same length")

    # Fast path: all channels homogeneous → standard kernel
    if len(set(zip(dists, links, strict=True))) == 1:
        return build_observation_kernel(
            dists[0],
            links[0],
            extra_params,
            manifest_cov=manifest_cov,
        )

    # Group channels by (dist, link)
    from collections import defaultdict

    groups: dict[tuple[DistributionFamily, LinkFunction], list[int]] = defaultdict(list)
    for ch_idx in range(n_manifest):
        groups[(dists[ch_idx], links[ch_idx])].append(ch_idx)

    # Build per-group kernels
    group_kernels: list[tuple[list[int], ObservationKernel]] = []
    for (dist, link), ch_indices in groups.items():
        kernel = build_observation_kernel(
            dist,
            link,
            _slice_observation_extra_params(extra_params, ch_indices),
            manifest_cov=(
                manifest_cov[jnp.ix_(jnp.asarray(ch_indices), jnp.asarray(ch_indices))]
                if manifest_cov is not None
                else None
            ),
        )
        group_kernels.append((ch_indices, kernel))

    # Compose emission_fn: sum per-group emission log-probs
    def composite_emission_fn(y_t, z_t, H, d, R, mask_t):
        total_ll = 0.0
        for ch_indices, kernel in group_kernels:
            idx = jnp.array(ch_indices)
            y_g = y_t[idx]
            H_g = H[idx, :]
            d_g = d[idx]
            R_g = R[jnp.ix_(idx, idx)]
            mask_g = mask_t[idx]
            total_ll = total_ll + kernel.emission_fn(y_g, z_t, H_g, d_g, R_g, mask_g)
        return total_ll

    # Compose emission_grad_hess_fn: sum per-group gradients and Hessians
    def composite_emission_grad_hess_fn(y_t, z_t, H, d, R, mask_t):
        D = z_t.shape[0]
        total_grad = jnp.zeros(D)
        total_hess = jnp.zeros((D, D))
        for ch_indices, kernel in group_kernels:
            idx = jnp.array(ch_indices)
            y_g = y_t[idx]
            H_g = H[idx, :]
            d_g = d[idx]
            R_g = R[jnp.ix_(idx, idx)]
            mask_g = mask_t[idx]
            g, neg_H = kernel.emission_grad_hess_fn(y_g, z_t, H_g, d_g, R_g, mask_g)
            total_grad = total_grad + g
            total_hess = total_hess + neg_H
        return total_grad, total_hess

    def composite_response_fn(eta: jnp.ndarray) -> jnp.ndarray:
        response = jnp.zeros_like(eta)
        for ch_indices, kernel in group_kernels:
            idx = jnp.array(ch_indices)
            response = response.at[idx].set(kernel.response_fn(eta[idx]))
        return response

    def composite_variance_fn(mean: jnp.ndarray) -> jnp.ndarray:
        variance = jnp.zeros((n_manifest, n_manifest), dtype=mean.dtype)
        for ch_indices, kernel in group_kernels:
            idx = jnp.array(ch_indices)
            variance = variance.at[jnp.ix_(idx, idx)].set(kernel.variance_fn(mean[idx]))
        return variance

    return ObservationKernel(
        emission_fn=composite_emission_fn,
        response_fn=composite_response_fn,
        variance_fn=composite_variance_fn,
        is_gaussian=False,  # heterogeneous is never purely Gaussian
        emission_grad_hess_fn=composite_emission_grad_hess_fn,
    )


def compile_measurement_semantics(
    manifest_dists: Sequence[DistributionFamily | str],
    *,
    manifest_cov: jnp.ndarray | None = None,
    extra_params: dict | None = None,
    manifest_links: Sequence[LinkFunction | str | None] | None = None,
    observation_support: ObservationSupportRuntime | None = None,
) -> MeasurementSemantics:
    """Compile observation kernels, mean-space likelihoods, and support semantics together."""
    from nof1_causal_lab.models.ssm.inference.targets.observation_families import (
        resolve_manifest_families_and_links,
    )
    from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
        compile_observation_operator,
    )

    n_manifest = len(manifest_dists)
    if manifest_cov is not None and int(manifest_cov.shape[0]) != n_manifest:
        raise ValueError(
            "manifest_cov width must match manifest_dists length: "
            f"{int(manifest_cov.shape[0])} vs {n_manifest}"
        )
    if observation_support is not None and len(observation_support.support_kinds) != n_manifest:
        raise ValueError(
            "observation_support width must match manifest_dists length: "
            f"{len(observation_support.support_kinds)} vs {n_manifest}"
        )

    dists, links = resolve_manifest_families_and_links(
        manifest_dists,
        manifest_links=manifest_links,
    )
    if len(set(zip(dists, links, strict=True))) == 1:
        obs_kernel = build_observation_kernel(
            dists[0],
            links[0],
            extra_params,
            manifest_cov=manifest_cov,
        )
    else:
        obs_kernel = build_composite_observation_kernel(
            dists,
            links,
            extra_params,
            manifest_cov=manifest_cov,
        )

    observation_operator = compile_observation_operator(observation_support)
    mean_log_prob_fn = None
    if observation_operator.requires_interval_summary_handling:
        interval_summary_indices = list(observation_operator.interval_summary_indices)
        interval_summary_idx = np.asarray(interval_summary_indices, dtype=np.int32)
        interval_summary_dists = [dists[idx] for idx in interval_summary_indices]
        if len(set(interval_summary_dists)) == 1:
            base_mean_log_prob_fn = get_mean_param_log_prob_fn(
                interval_summary_dists[0], extra_params
            )
        else:
            base_mean_log_prob_fn = build_composite_mean_log_prob_fn(
                [dist.value for dist in interval_summary_dists],
                _slice_observation_extra_params(extra_params, interval_summary_indices),
            )

        def mean_log_prob_fn(y_t, mean_t, R, obs_mask_t):
            y_interval_summary = y_t[interval_summary_idx]
            mean_interval_summary = mean_t[interval_summary_idx]
            mask_interval_summary = obs_mask_t[interval_summary_idx]
            R_interval_summary = R[np.ix_(interval_summary_idx, interval_summary_idx)]
            return base_mean_log_prob_fn(
                y_interval_summary,
                mean_interval_summary,
                R_interval_summary,
                mask_interval_summary,
            )

    return MeasurementSemantics(
        obs_kernel=obs_kernel,
        mean_log_prob_fn=mean_log_prob_fn,
        observation_operator=observation_operator,
        manifest_dists=tuple(dists),
        manifest_links=tuple(links),
    )
