"""Rao-Blackwell partition artifacts for runtime-conditioned SSM inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter

RBPFMode = Literal["none", "independent", "conditional"]
SUPPORTED_RBPF_MODES: tuple[RBPFMode, ...] = ("none", "independent", "conditional")


@dataclass(frozen=True)
class RBPFPartitionSpec:
    """Static latent partition consumed by RBPF-conditioned runtimes."""

    carried_latent_indices: tuple[int, ...]
    marginalized_latent_indices: tuple[int, ...]

    @property
    def is_full_path(self) -> bool:
        return len(self.marginalized_latent_indices) == 0


@dataclass(frozen=True)
class RBPFObservationPlan:
    """Manifest rows consumed by a collapsed Gaussian RBPF block."""

    channel_mask: jnp.ndarray
    gaussian_channel_mask: jnp.ndarray
    polya_gamma_channel_mask: jnp.ndarray
    structure: str

    @property
    def enabled(self) -> bool:
        return bool(np.any(np.asarray(self.channel_mask)))


class RBPFMarginalContext(NamedTuple):
    """Linear-Gaussian subsystem integrated out conditional on sampled parameters."""

    Ad_cc: jnp.ndarray
    Ad_mm: jnp.ndarray
    Ad_mc: jnp.ndarray
    Q_cc: jnp.ndarray
    Q_mm: jnp.ndarray
    Q_mc: jnp.ndarray
    cd_c: jnp.ndarray
    cd_m: jnp.ndarray
    init_mean_c: jnp.ndarray
    init_mean_m: jnp.ndarray
    init_cov_cc: jnp.ndarray
    init_cov_mm: jnp.ndarray
    init_cov_mc: jnp.ndarray
    H_m: jnp.ndarray
    H_c: jnp.ndarray
    d_meas: jnp.ndarray
    R: jnp.ndarray
    observation_indices: jnp.ndarray
    gaussian_row_mask: jnp.ndarray
    polya_gamma_row_mask: jnp.ndarray


class RBPFMarginalFilterState(NamedTuple):
    """Filtered marginalized-state moments for one carried particle/path prefix."""

    mean: jnp.ndarray
    cov: jnp.ndarray


def normalize_rbpf_mode(rbpf_mode: str) -> RBPFMode:
    """Normalize and validate the public RBPF runtime-conditioning mode."""
    mode = str(rbpf_mode).strip().lower()
    if mode not in SUPPORTED_RBPF_MODES:
        raise ValueError(
            f"Unsupported rbpf_mode {rbpf_mode!r}. "
            f"Supported: {', '.join(repr(mode) for mode in SUPPORTED_RBPF_MODES)}."
        )
    return cast("RBPFMode", mode)


def full_path_rbpf_partition(n_latent: int) -> RBPFPartitionSpec:
    """Return the identity partition: every latent dimension remains sampled."""
    if n_latent < 1:
        raise ValueError(f"n_latent must be positive, got {n_latent}.")
    return RBPFPartitionSpec(
        carried_latent_indices=tuple(range(int(n_latent))),
        marginalized_latent_indices=(),
    )


def build_rbpf_partition(
    n_latent: int,
    marginalized_latent_indices: tuple[int, ...] | list[int] | None,
) -> RBPFPartitionSpec:
    """Build a static partition from marginalized latent indices."""
    if marginalized_latent_indices is None:
        return full_path_rbpf_partition(n_latent)
    marginalized = tuple(int(idx) for idx in marginalized_latent_indices)
    marginalized_set = set(marginalized)
    carried = tuple(idx for idx in range(int(n_latent)) if idx not in marginalized_set)
    partition = RBPFPartitionSpec(
        carried_latent_indices=carried,
        marginalized_latent_indices=marginalized,
    )
    validate_rbpf_partition(partition, n_latent=n_latent)
    return partition


def validate_rbpf_partition(partition: RBPFPartitionSpec, *, n_latent: int) -> None:
    """Validate a partition before attaching it to a conditioned runtime."""
    all_indices = partition.carried_latent_indices + partition.marginalized_latent_indices
    if sorted(all_indices) != list(range(n_latent)):
        raise ValueError(
            "RBPF partition must cover each latent index exactly once; "
            f"got carried={partition.carried_latent_indices}, "
            f"marginalized={partition.marginalized_latent_indices}, n_latent={n_latent}."
        )
    if not partition.carried_latent_indices:
        raise ValueError("RBPF requires at least one carried latent dimension.")


def _enum_value_lower(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _is_active_entry(free_support: np.ndarray, template: np.ndarray) -> np.ndarray:
    return np.asarray(free_support, dtype=bool) | ~np.isclose(np.asarray(template), 0.0)


def _cross_block_active(active: np.ndarray, left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return bool(np.any(active[np.ix_(left, right)])) or bool(np.any(active[np.ix_(right, left)]))


def _latent_group(idx: int, partition: RBPFPartitionSpec) -> str:
    if idx in partition.carried_latent_indices:
        return "carried"
    if idx in partition.marginalized_latent_indices:
        return "marginalized"
    raise ValueError(f"Latent index {idx} is not covered by RBPF partition {partition}.")


def _validate_rbpf_dynamics_components(spec, partition: RBPFPartitionSpec) -> bool:
    conditional = False
    for component in spec.dynamics_spec.components:
        if hasattr(component, "source") and hasattr(component, "target"):
            source_group = _latent_group(int(component.source), partition)
            target_group = _latent_group(int(component.target), partition)
            if source_group == "marginalized" and target_group == "carried":
                raise NotImplementedError(
                    "RBPF marginalization requires carried dynamics not to depend on "
                    f"marginalized latents; component {component!r} violates this."
                )
            if source_group == "carried" and target_group == "marginalized":
                if type(component).__name__ != "LinearEdgeSpec":
                    raise NotImplementedError(
                        "RBPF conditional marginalized dynamics currently supports only linear "
                        f"carried->marginalized edges; got {component!r}."
                    )
                conditional = True
        if (
            hasattr(component, "source_a")
            and hasattr(component, "source_b")
            and hasattr(component, "target")
        ):
            groups = {
                _latent_group(int(component.source_a), partition),
                _latent_group(int(component.source_b), partition),
                _latent_group(int(component.target), partition),
            }
            if len(groups) != 1:
                raise NotImplementedError(
                    "RBPF conditional marginalized dynamics does not support multiplicative "
                    "components that cross the carried/marginalized partition; "
                    f"component {component!r} crosses the carried/marginalized partition."
                )
    return conditional


def build_gaussian_rbpf_observation_plan(
    spec,
    partition: RBPFPartitionSpec,
    manifest_links: list[LinkFunction],
    polya_gamma_channel_mask: jnp.ndarray,
) -> RBPFObservationPlan:
    """Validate and identify rows for the Gaussian RBPF adapter."""
    validate_rbpf_partition(partition, n_latent=spec.n_latent)
    empty_mask = jnp.zeros((spec.n_manifest,), dtype=bool)
    if partition.is_full_path:
        return RBPFObservationPlan(
            channel_mask=empty_mask,
            gaussian_channel_mask=empty_mask,
            polya_gamma_channel_mask=empty_mask,
            structure="none",
        )

    carried = partition.carried_latent_indices
    marginalized = partition.marginalized_latent_indices
    conditional = _validate_rbpf_dynamics_components(spec, partition)

    diffusion_active = _is_active_entry(
        spec.diffusion_block.diffusion_chol_support,
        spec.diffusion_block.diffusion_chol_template,
    )
    diffusion_cross_active = _cross_block_active(diffusion_active, carried, marginalized)
    if diffusion_cross_active:
        raise NotImplementedError(
            "RBPF marginalization currently requires block-diagonal instantaneous process diffusion."
        )

    t0_corr_support = np.asarray(spec.t0_chol_block.correlation_support, dtype=bool)
    t0_corr_active = t0_corr_support | t0_corr_support.T
    t0_base_cov = np.asarray(spec.t0_chol_block.base_cov)
    t0_base_active = ~np.isclose(t0_base_cov, 0.0)
    if _cross_block_active(t0_corr_active | t0_base_active, carried, marginalized):
        raise NotImplementedError(
            "RBPF marginalization currently requires block-diagonal initial-state covariance."
        )

    loading_active = _is_active_entry(
        spec.lambda_block.free_support,
        spec.lambda_block.template,
    )
    uses_carried = np.any(loading_active[:, carried], axis=1)
    uses_marginalized = np.any(loading_active[:, marginalized], axis=1)
    pg_channels = np.asarray(polya_gamma_channel_mask, dtype=bool)
    if pg_channels.shape != (spec.n_manifest,):
        raise ValueError(
            "polya_gamma_channel_mask must have shape "
            f"({spec.n_manifest},), got {pg_channels.shape}."
        )
    gaussian_channels = np.zeros((spec.n_manifest,), dtype=bool)
    rbpf_pg_channels = np.zeros((spec.n_manifest,), dtype=bool)
    for channel_idx in np.where(uses_marginalized)[0].tolist():
        dist_value = _enum_value_lower(spec.manifest_dists[channel_idx])
        link_value = _enum_value_lower(manifest_links[channel_idx])
        is_gaussian_identity = dist_value == _enum_value_lower(
            DistributionFamily.GAUSSIAN
        ) and link_value == _enum_value_lower(LinkFunction.IDENTITY)
        if is_gaussian_identity:
            gaussian_channels[channel_idx] = True
            continue
        if bool(pg_channels[channel_idx]):
            rbpf_pg_channels[channel_idx] = True
            continue
        raise NotImplementedError(
            "RBPF marginalization currently consumes Gaussian identity rows or "
            "PG-conditioned affine-logit rows; "
            f"channel {channel_idx} has distribution={dist_value}, link={link_value}."
        )
    marginal_channels = gaussian_channels | rbpf_pg_channels
    if bool(np.any(rbpf_pg_channels)):
        manifest_chol = np.asarray(spec.manifest_chol_block.template)
        manifest_cov = manifest_chol @ manifest_chol.T
        pg_indices = np.where(rbpf_pg_channels)[0]
        gaussian_indices = np.where(gaussian_channels)[0]
        if (
            pg_indices.size
            and gaussian_indices.size
            and bool(np.any(~np.isclose(manifest_cov[np.ix_(pg_indices, gaussian_indices)], 0.0)))
        ):
            raise NotImplementedError(
                "RBPF PG-conditioned rows require zero manifest-noise covariance "
                "with Gaussian RBPF rows."
            )
    conditional = conditional or bool(np.any(uses_carried & uses_marginalized))

    return RBPFObservationPlan(
        channel_mask=jnp.asarray(marginal_channels, dtype=bool),
        gaussian_channel_mask=jnp.asarray(gaussian_channels, dtype=bool),
        polya_gamma_channel_mask=jnp.asarray(rbpf_pg_channels, dtype=bool),
        structure="conditional" if conditional else "independent",
    )


def validate_rbpf_mode(
    rbpf_mode: RBPFMode,
    partition: RBPFPartitionSpec,
    observation_plan: RBPFObservationPlan,
) -> None:
    """Require the explicit requested RBPF mode to match the validated structure."""
    if rbpf_mode == "none":
        if not partition.is_full_path:
            raise ValueError(
                "rbpf_mode='none' cannot be combined with rbpf_marginalized_latent_indices."
            )
        if observation_plan.structure != "none":
            raise ValueError("rbpf_mode='none' requires an unpartitioned full-path runtime.")
        return

    if partition.is_full_path:
        raise ValueError(f"rbpf_mode={rbpf_mode!r} requires rbpf_marginalized_latent_indices.")
    if observation_plan.structure != rbpf_mode:
        raise ValueError(
            f"rbpf_mode={rbpf_mode!r} does not match the validated RBPF structure "
            f"{observation_plan.structure!r}. Choose the matching mode explicitly."
        )


def mask_rbpf_observations(plan: RBPFObservationPlan, observations: jnp.ndarray) -> jnp.ndarray:
    """Remove RBPF-consumed observations from the carried-state residual likelihood."""
    return jnp.where(plan.channel_mask[None, :], jnp.nan, observations)


def _take_square(mats: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(jnp.take(mats, indices, axis=-2), indices, axis=-1)


def _take_rect(mats: jnp.ndarray, rows: jnp.ndarray, cols: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(jnp.take(mats, rows, axis=-2), cols, axis=-1)


def build_rbpf_marginal_context(
    *,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    H: jnp.ndarray,
    d_meas: jnp.ndarray,
    R: jnp.ndarray,
    partition: RBPFPartitionSpec,
    observation_plan: RBPFObservationPlan,
) -> RBPFMarginalContext:
    """Project the full linear-Gaussian context onto carried/marginalized blocks."""
    carried = jnp.asarray(partition.carried_latent_indices, dtype=jnp.int32)
    marginalized = jnp.asarray(partition.marginalized_latent_indices, dtype=jnp.int32)
    observation_indices = jnp.nonzero(
        observation_plan.channel_mask,
        size=int(np.sum(np.asarray(observation_plan.channel_mask))),
    )[0].astype(jnp.int32)
    gaussian_row_mask = jnp.take(observation_plan.gaussian_channel_mask, observation_indices)
    pg_row_mask = jnp.take(observation_plan.polya_gamma_channel_mask, observation_indices)
    init_pred_mean = Ad[0] @ init_mean + cd[0]
    init_pred_cov = symmetrize_with_jitter(Ad[0] @ init_cov @ Ad[0].T + Qd[0], jitter=0.0)
    return RBPFMarginalContext(
        Ad_cc=_take_square(Ad, carried),
        Ad_mm=_take_square(Ad, marginalized),
        Ad_mc=_take_rect(Ad, marginalized, carried),
        Q_cc=_take_square(Qd, carried),
        Q_mm=_take_square(Qd, marginalized),
        Q_mc=_take_rect(Qd, marginalized, carried),
        cd_c=jnp.take(cd, carried, axis=-1),
        cd_m=jnp.take(cd, marginalized, axis=-1),
        init_mean_c=jnp.take(init_pred_mean, carried, axis=-1),
        init_mean_m=jnp.take(init_pred_mean, marginalized, axis=-1),
        init_cov_cc=_take_square(init_pred_cov, carried),
        init_cov_mm=_take_square(init_pred_cov, marginalized),
        init_cov_mc=_take_rect(init_pred_cov, marginalized, carried),
        H_m=jnp.take(jnp.take(H, observation_indices, axis=0), marginalized, axis=1),
        H_c=jnp.take(jnp.take(H, observation_indices, axis=0), carried, axis=1),
        d_meas=jnp.take(d_meas, observation_indices, axis=0),
        R=_take_square(R, observation_indices),
        observation_indices=observation_indices,
        gaussian_row_mask=gaussian_row_mask,
        polya_gamma_row_mask=pg_row_mask,
    )


def reduce_context_to_carried(
    *,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    H: jnp.ndarray,
    partition: RBPFPartitionSpec,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project state-transition and loading matrices onto carried dimensions."""
    carried = jnp.asarray(partition.carried_latent_indices, dtype=jnp.int32)
    return (
        _take_square(Ad, carried),
        _take_square(Qd, carried),
        jnp.take(cd, carried, axis=-1),
        jnp.take(init_mean, carried, axis=-1),
        _take_square(init_cov, carried),
        jnp.take(H, carried, axis=1),
    )


def _solve_spd(mat: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    return jla.solve(mat, rhs, assume_a="pos")


def _condition_marginal_initial(
    marginal_context: RBPFMarginalContext,
    carried_t: jnp.ndarray,
    *,
    jitter: float,
) -> RBPFMarginalFilterState:
    cov_cc = symmetrize_with_jitter(marginal_context.init_cov_cc, jitter=jitter)
    diff_c = carried_t - marginal_context.init_mean_c
    mean = marginal_context.init_mean_m + marginal_context.init_cov_mc @ _solve_spd(cov_cc, diff_c)
    cov = marginal_context.init_cov_mm - marginal_context.init_cov_mc @ _solve_spd(
        cov_cc,
        marginal_context.init_cov_mc.T,
    )
    return RBPFMarginalFilterState(
        mean=mean,
        cov=symmetrize_with_jitter(cov, jitter=jitter),
    )


def _rbpf_observation_at(
    marginal_context: RBPFMarginalContext,
    observations: jnp.ndarray,
    observation_auxiliary,
    time_idx: jnp.ndarray,
    *,
    dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    y_raw = jnp.take(observations[time_idx], marginal_context.observation_indices, axis=0)
    y_raw = jnp.nan_to_num(y_raw, nan=0.0).astype(dtype)
    row_pg_mask = marginal_context.polya_gamma_row_mask.astype(dtype)
    row_gaussian_mask = marginal_context.gaussian_row_mask.astype(dtype)
    if observation_auxiliary is None:
        omega_rows = jnp.ones_like(y_raw)
        kappa_rows = y_raw
        offset_rows = jnp.zeros_like(y_raw)
    else:
        omega_t = jnp.take(
            observation_auxiliary.omega[time_idx],
            marginal_context.observation_indices,
            axis=0,
        )
        kappa_t = jnp.take(
            observation_auxiliary.kappa[time_idx],
            marginal_context.observation_indices,
            axis=0,
        )
        offset_t = jnp.take(
            observation_auxiliary.linear_offset[time_idx],
            marginal_context.observation_indices,
            axis=0,
        )
        omega_rows = omega_t.astype(dtype)
        kappa_rows = kappa_t.astype(dtype)
        offset_rows = offset_t.astype(dtype)

    omega_safe = jnp.maximum(omega_rows, jnp.asarray(1e-8, dtype=dtype))
    pg_y = kappa_rows / omega_safe - offset_rows
    y_t = jnp.where(marginal_context.polya_gamma_row_mask, pg_y, y_raw)
    gaussian_outer = row_gaussian_mask[:, None] * row_gaussian_mask[None, :]
    R_gaussian = marginal_context.R.astype(dtype) * gaussian_outer
    R_pg = jnp.diag(row_pg_mask / omega_safe)
    return y_t, R_gaussian + R_pg


def _rbpf_gaussian_update(
    marginal_context: RBPFMarginalContext,
    filter_state: RBPFMarginalFilterState,
    observations_t: jnp.ndarray,
    observation_cov_t: jnp.ndarray,
    carried_t: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[RBPFMarginalFilterState, jnp.ndarray]:
    H_m = marginal_context.H_m
    if H_m.shape[0] == 0:
        return filter_state, jnp.asarray(0.0, dtype=carried_t.dtype)
    H_c = marginal_context.H_c
    resid = observations_t - (H_m @ filter_state.mean + H_c @ carried_t + marginal_context.d_meas)
    innovation_cov = symmetrize_with_jitter(
        H_m @ filter_state.cov @ H_m.T + observation_cov_t,
        jitter=jitter,
    )
    chol = jnp.linalg.cholesky(innovation_cov)
    whitened = jla.solve_triangular(chol, resid, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diag(chol), 1e-12)))
    loglik = -0.5 * (
        observations_t.shape[-1] * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=observations_t.dtype))
        + logdet
        + whitened @ whitened
    )
    PHt = filter_state.cov @ H_m.T
    gain_t = jla.solve_triangular(chol, PHt.T, lower=True)
    gain = jla.solve_triangular(chol.T, gain_t, lower=False).T
    updated_mean = filter_state.mean + gain @ resid
    updated_cov = symmetrize_with_jitter(
        filter_state.cov - gain @ H_m @ filter_state.cov,
        jitter=jitter,
    )
    return RBPFMarginalFilterState(mean=updated_mean, cov=updated_cov), loglik


def rbpf_initial_filter_update(
    marginal_context: RBPFMarginalContext | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_t: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> tuple[RBPFMarginalFilterState | None, jnp.ndarray]:
    """Initialize and update marginalized filter at t=0 for one carried state."""
    if marginal_context is None or marginal_context.H_m.shape[0] == 0:
        return None, jnp.asarray(0.0, dtype=carried_t.dtype)
    y0, R0 = _rbpf_observation_at(
        marginal_context,
        observations,
        observation_auxiliary,
        jnp.asarray(0, dtype=jnp.int32),
        dtype=carried_t.dtype,
    )
    initial_state = _condition_marginal_initial(marginal_context, carried_t, jitter=jitter)
    return _rbpf_gaussian_update(
        marginal_context,
        initial_state,
        y0,
        R0,
        carried_t,
        jitter=jitter,
    )


def rbpf_step_filter_update(
    marginal_context: RBPFMarginalContext | None,
    previous_filter_state: RBPFMarginalFilterState | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_prev: jnp.ndarray,
    carried_t: jnp.ndarray,
    time_idx: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> tuple[RBPFMarginalFilterState | None, jnp.ndarray]:
    """Predict/update marginalized filter for one carried transition."""
    if marginal_context is None or marginal_context.H_m.shape[0] == 0:
        return None, jnp.asarray(0.0, dtype=carried_t.dtype)
    assert previous_filter_state is not None
    Ad_cc_t = marginal_context.Ad_cc[time_idx]
    Ad_mm_t = marginal_context.Ad_mm[time_idx]
    Ad_mc_t = marginal_context.Ad_mc[time_idx]
    Q_cc_t = symmetrize_with_jitter(marginal_context.Q_cc[time_idx], jitter=jitter)
    Q_mm_t = marginal_context.Q_mm[time_idx]
    Q_mc_t = marginal_context.Q_mc[time_idx]
    cd_c_t = marginal_context.cd_c[time_idx]
    cd_m_t = marginal_context.cd_m[time_idx]

    carried_innovation = carried_t - (Ad_cc_t @ carried_prev + cd_c_t)
    noise_mean = Q_mc_t @ _solve_spd(Q_cc_t, carried_innovation)
    noise_cov = Q_mm_t - Q_mc_t @ _solve_spd(Q_cc_t, Q_mc_t.T)
    pred_mean = Ad_mm_t @ previous_filter_state.mean + Ad_mc_t @ carried_prev + cd_m_t + noise_mean
    pred_cov = symmetrize_with_jitter(
        Ad_mm_t @ previous_filter_state.cov @ Ad_mm_t.T + noise_cov,
        jitter=jitter,
    )
    y_t, R_t = _rbpf_observation_at(
        marginal_context,
        observations,
        observation_auxiliary,
        time_idx,
        dtype=carried_t.dtype,
    )
    return _rbpf_gaussian_update(
        marginal_context,
        RBPFMarginalFilterState(mean=pred_mean, cov=pred_cov),
        y_t,
        R_t,
        carried_t,
        jitter=jitter,
    )


def rbpf_marginal_log_likelihood(
    marginal_context: RBPFMarginalContext | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_trajectory: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """Kalman marginal likelihood of the marginalized subsystem conditional on carried path."""
    if marginal_context is None:
        return jnp.asarray(0.0, dtype=carried_trajectory.dtype)
    if marginal_context.H_m.shape[0] == 0:
        return jnp.asarray(0.0, dtype=carried_trajectory.dtype)

    filter0, init_loglik = rbpf_initial_filter_update(
        marginal_context,
        observations,
        observation_auxiliary,
        carried_trajectory[0],
        jitter=jitter,
    )

    def _step(carry, inputs):
        previous_filter, carried_prev = carry
        carried_t, time_idx = inputs
        next_filter, loglik = rbpf_step_filter_update(
            marginal_context,
            previous_filter,
            observations,
            observation_auxiliary,
            carried_prev,
            carried_t,
            time_idx,
            jitter=jitter,
        )
        return (next_filter, carried_t), loglik

    if carried_trajectory.shape[0] == 1:
        return jnp.asarray(init_loglik, dtype=carried_trajectory.dtype)
    _, loglik_rest = jax.lax.scan(
        _step,
        (filter0, carried_trajectory[0]),
        (
            carried_trajectory[1:],
            jnp.arange(1, carried_trajectory.shape[0], dtype=jnp.int32),
        ),
    )
    return jnp.asarray(init_loglik + jnp.sum(loglik_rest), dtype=carried_trajectory.dtype)


def rbpf_marginal_log_likelihoods(
    marginal_context: RBPFMarginalContext | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_trajectory: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """Per-time Kalman marginal likelihoods conditional on a carried path."""
    if marginal_context is None or marginal_context.H_m.shape[0] == 0:
        return jnp.zeros((carried_trajectory.shape[0],), dtype=carried_trajectory.dtype)
    filter0, init_loglik = rbpf_initial_filter_update(
        marginal_context,
        observations,
        observation_auxiliary,
        carried_trajectory[0],
        jitter=jitter,
    )

    def _step(carry, inputs):
        previous_filter, carried_prev = carry
        carried_t, time_idx = inputs
        next_filter, loglik = rbpf_step_filter_update(
            marginal_context,
            previous_filter,
            observations,
            observation_auxiliary,
            carried_prev,
            carried_t,
            time_idx,
            jitter=jitter,
        )
        return (next_filter, carried_t), loglik

    if carried_trajectory.shape[0] == 1:
        return jnp.reshape(init_loglik, (1,))
    _, loglik_rest = jax.lax.scan(
        _step,
        (filter0, carried_trajectory[0]),
        (
            carried_trajectory[1:],
            jnp.arange(1, carried_trajectory.shape[0], dtype=jnp.int32),
        ),
    )
    return jnp.concatenate([jnp.reshape(init_loglik, (1,)), loglik_rest])


def _sample_gaussian(key: jax.Array, mean: jnp.ndarray, cov: jnp.ndarray, *, jitter: float):
    chol = jnp.linalg.cholesky(symmetrize_with_jitter(cov, jitter=jitter))
    eps = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    return mean + chol @ eps


def _rbpf_transition_condition(
    marginal_context: RBPFMarginalContext,
    previous_filter_state: RBPFMarginalFilterState,
    carried_prev: jnp.ndarray,
    carried_t: jnp.ndarray,
    time_idx: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Ad_cc_t = marginal_context.Ad_cc[time_idx]
    Ad_mm_t = marginal_context.Ad_mm[time_idx]
    Ad_mc_t = marginal_context.Ad_mc[time_idx]
    Q_cc_t = symmetrize_with_jitter(marginal_context.Q_cc[time_idx], jitter=jitter)
    Q_mm_t = marginal_context.Q_mm[time_idx]
    Q_mc_t = marginal_context.Q_mc[time_idx]
    cd_c_t = marginal_context.cd_c[time_idx]
    cd_m_t = marginal_context.cd_m[time_idx]

    carried_innovation = carried_t - (Ad_cc_t @ carried_prev + cd_c_t)
    noise_mean = Q_mc_t @ _solve_spd(Q_cc_t, carried_innovation)
    noise_cov = Q_mm_t - Q_mc_t @ _solve_spd(Q_cc_t, Q_mc_t.T)
    transition_offset = Ad_mc_t @ carried_prev + cd_m_t + noise_mean
    pred_mean = Ad_mm_t @ previous_filter_state.mean + transition_offset
    pred_cov = symmetrize_with_jitter(
        Ad_mm_t @ previous_filter_state.cov @ Ad_mm_t.T + noise_cov,
        jitter=jitter,
    )
    return pred_mean, pred_cov, transition_offset


def sample_rbpf_marginal_trajectory(
    key: jax.Array,
    marginal_context: RBPFMarginalContext | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_trajectory: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """Sample marginalized latent path conditional on carried path and Gaussian/PG rows."""
    if marginal_context is None:
        return jnp.zeros((carried_trajectory.shape[0], 0), dtype=carried_trajectory.dtype)
    n_marginalized = int(marginal_context.Ad_mm.shape[-1])
    if n_marginalized == 0:
        return jnp.zeros((carried_trajectory.shape[0], 0), dtype=carried_trajectory.dtype)

    filter0, _init_loglik = rbpf_initial_filter_update(
        marginal_context,
        observations,
        observation_auxiliary,
        carried_trajectory[0],
        jitter=jitter,
    )
    assert filter0 is not None

    def _forward_step(carry, inputs):
        previous_filter, carried_prev = carry
        carried_t, time_idx = inputs
        pred_mean, pred_cov, _offset = _rbpf_transition_condition(
            marginal_context,
            previous_filter,
            carried_prev,
            carried_t,
            time_idx,
            jitter=jitter,
        )
        y_t, R_t = _rbpf_observation_at(
            marginal_context,
            observations,
            observation_auxiliary,
            time_idx,
            dtype=carried_t.dtype,
        )
        next_filter, _loglik = _rbpf_gaussian_update(
            marginal_context,
            RBPFMarginalFilterState(mean=pred_mean, cov=pred_cov),
            y_t,
            R_t,
            carried_t,
            jitter=jitter,
        )
        return (next_filter, carried_t), next_filter

    if carried_trajectory.shape[0] == 1:
        filter_means = filter0.mean[None, :]
        filter_covs = filter0.cov[None, :, :]
    else:
        _, filter_rest = jax.lax.scan(
            _forward_step,
            (filter0, carried_trajectory[0]),
            (
                carried_trajectory[1:],
                jnp.arange(1, carried_trajectory.shape[0], dtype=jnp.int32),
            ),
        )
        filter_means = jnp.concatenate([filter0.mean[None, :], filter_rest.mean], axis=0)
        filter_covs = jnp.concatenate([filter0.cov[None, :, :], filter_rest.cov], axis=0)

    keys = jax.random.split(key, carried_trajectory.shape[0])
    final_sample = _sample_gaussian(
        keys[-1],
        filter_means[-1],
        filter_covs[-1],
        jitter=jitter,
    )

    def _backward_step(next_sample, inputs):
        filter_mean_t, filter_cov_t, carried_t, carried_next, time_idx, sample_key = inputs
        filter_state_t = RBPFMarginalFilterState(mean=filter_mean_t, cov=filter_cov_t)
        pred_mean, pred_cov, transition_offset = _rbpf_transition_condition(
            marginal_context,
            filter_state_t,
            carried_t,
            carried_next,
            time_idx,
            jitter=jitter,
        )
        F_t = marginal_context.Ad_mm[time_idx]
        smoother_gain = filter_cov_t @ F_t.T @ _solve_spd(pred_cov, jnp.eye(pred_cov.shape[0]))
        smooth_mean = filter_mean_t + smoother_gain @ (next_sample - pred_mean)
        smooth_cov = filter_cov_t - smoother_gain @ pred_cov @ smoother_gain.T
        del transition_offset
        sample_t = _sample_gaussian(sample_key, smooth_mean, smooth_cov, jitter=jitter)
        return sample_t, sample_t

    if carried_trajectory.shape[0] == 1:
        return final_sample[None, :]
    _, reversed_samples = jax.lax.scan(
        _backward_step,
        final_sample,
        (
            filter_means[:-1][::-1],
            filter_covs[:-1][::-1],
            carried_trajectory[:-1][::-1],
            carried_trajectory[1:][::-1],
            jnp.arange(1, carried_trajectory.shape[0], dtype=jnp.int32)[::-1],
            keys[:-1][::-1],
        ),
    )
    return jnp.concatenate([reversed_samples[::-1], final_sample[None, :]], axis=0)
