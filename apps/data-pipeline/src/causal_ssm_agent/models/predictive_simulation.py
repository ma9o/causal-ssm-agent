"""Shared predictive observation simulation for prior and posterior workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import lax, vmap

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.constants import MIN_DT
from causal_ssm_agent.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stable_cholesky,
)
from causal_ssm_agent.models.ssm.discretization import discretize_system_batched
from causal_ssm_agent.models.ssm.inference.targets.emissions import (
    build_predictive_observation_sampler,
)
from causal_ssm_agent.models.ssm.inference.targets.kernels import (
    build_composite_observation_kernel,
    build_observation_kernel,
    build_transition_kernel,
    compile_transition_semantics,
)
from causal_ssm_agent.models.ssm.inference.targets.observation_families import (
    any_family_needs_level_metadata,
    resolve_manifest_families_and_links,
)
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    compile_observation_operator,
)
from causal_ssm_agent.orchestrator.schemas_model import LinkFunction

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

logger = get_prefect_logger(__name__)


class PredictiveObservationMeanOverflow(RuntimeError):
    """Raised when a log-link predictive mean exceeds finite numeric range."""

    def __init__(
        self,
        *,
        bad_manifest_names: tuple[str, ...],
        manifest_indices: tuple[int, ...],
        failing_draw_indices: tuple[int, ...],
        first_bad_time_index: int,
        max_linear_predictor: float,
        overflow_threshold: float,
    ) -> None:
        self.bad_manifest_names = bad_manifest_names
        self.manifest_indices = manifest_indices
        self.failing_draw_indices = failing_draw_indices
        self.first_bad_time_index = first_bad_time_index
        self.max_linear_predictor = max_linear_predictor
        self.overflow_threshold = overflow_threshold
        manifest_summary = ", ".join(bad_manifest_names) if bad_manifest_names else "unknown"
        super().__init__(
            "Predictive log-link mean overflow before observation sampling: "
            f"linear predictor exceeded the finite exp range for {manifest_summary} "
            f"(max eta={max_linear_predictor:.2f}, threshold={overflow_threshold:.2f}, "
            f"first bad time index={first_bad_time_index})."
        )


def _broadcast_draw_param(
    value: jnp.ndarray | None,
    n_use: int,
    indices: jnp.ndarray,
) -> jnp.ndarray | None:
    if value is None:
        return None
    if value.ndim == 0:
        return jnp.broadcast_to(value, (n_use,))
    if value.shape[0] == n_use:
        return value
    if value.shape[0] >= int(indices[-1]) + 1:
        return value[indices]
    return jnp.broadcast_to(value, (n_use, *value.shape))


def _linear_predictors_from_latent_trajectory(
    latent_trajectory: jnp.ndarray,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
) -> jnp.ndarray:
    """Map a latent trajectory to observation linear predictors."""
    return jax.vmap(lambda eta_t: lambda_mat @ eta_t + manifest_means)(latent_trajectory)


def _simulate_latent_trajectory(
    drift: jnp.ndarray,
    diffusion_chol: jnp.ndarray,
    cint: jnp.ndarray | None,
    t0_mean: jnp.ndarray,
    t0_chol: jnp.ndarray,
    transition_dt_array: jnp.ndarray,
    n_timepoints: int,
    rng_key: jax.Array,
    transition_semantics,
    proc_df: float | jnp.ndarray = 5.0,
) -> jnp.ndarray:
    """Simulate one latent trajectory from the continuous-time dynamics."""
    n_latent = drift.shape[0]
    T = n_timepoints

    key_init, key_proc = random.split(rng_key)
    eta_0 = t0_mean + t0_chol @ random.normal(key_init, (n_latent,))
    if T == 1:
        return eta_0[None, :]

    diffusion_cov = diffusion_chol @ diffusion_chol.T
    Ad, Qd, cd = discretize_system_batched(drift, diffusion_cov, cint, transition_dt_array)
    if cd is None:
        cd = jnp.zeros((T - 1, n_latent))

    proc_keys = random.split(key_proc, T - 1)
    transition_kernel = build_transition_kernel(transition_semantics, {"proc_df": proc_df})

    def scan_fn(eta_prev, inputs):
        Ad_t, Qd_t, cd_t, pkey = inputs
        Qd_chol = stable_cholesky(Qd_t)
        noise = transition_kernel.sample_noise_fn(pkey, Qd_chol)
        eta_t = Ad_t @ eta_prev + cd_t + noise
        return eta_t, eta_t

    _, eta_rest = lax.scan(scan_fn, eta_0, (Ad, Qd, cd, proc_keys))
    return jnp.concatenate((eta_0[None, :], eta_rest), axis=0)


def _build_response_kernel(
    manifest_dist: str,
    manifest_dists: list[str] | None,
    manifest_links: list[str] | None,
    n_manifest: int,
    extra_params: dict | None,
):
    """Build the response-space observation kernel for one posterior draw."""
    dists, links = resolve_manifest_families_and_links(
        manifest_dist,
        n_manifest,
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
    )
    if len(set(zip(dists, links))) == 1:
        return build_observation_kernel(dists[0], links[0], extra_params)
    return build_composite_observation_kernel(dists, links, extra_params)


def _slice_extra_params_for_indices(
    extra_params: dict | None,
    indices: list[int],
) -> dict | None:
    """Slice per-channel extra params down to a manifest subset."""
    if extra_params is None:
        return None

    sliced: dict = {}
    idx = jnp.asarray(indices, dtype=jnp.int32)
    for key, value in extra_params.items():
        if hasattr(value, "ndim") and hasattr(value, "shape") and value.ndim >= 1:
            try:
                if value.shape[0] >= len(indices):
                    sliced[key] = value[idx]
                    continue
            except TypeError:
                logger.warning("Unexpected type for extra parameter %r during slicing", key)
        sliced[key] = value
    return sliced


def _resolve_effective_observation_mask(
    target_shape: tuple[int, ...],
    semantic_mask: jnp.ndarray | None,
    observation_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Return the explicit emitted-observation mask for one simulated draw."""
    if len(target_shape) == 2:
        mask_shape = target_shape
    else:
        mask_shape = target_shape[1:]
    effective_mask = jnp.ones(mask_shape, dtype=bool)
    if semantic_mask is not None:
        effective_mask = effective_mask & (semantic_mask > 0.5)
    if observation_mask is not None:
        effective_mask = effective_mask & observation_mask.astype(bool)
    return effective_mask


def _apply_observation_mask(
    y_sim: jnp.ndarray,
    semantic_mask: jnp.ndarray | None,
    observation_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Set structurally absent observations to NaN."""
    effective_mask = _resolve_effective_observation_mask(
        y_sim.shape,
        semantic_mask,
        observation_mask,
    )
    if y_sim.ndim == 2:
        return jnp.where(effective_mask, y_sim, jnp.nan)
    return jnp.where(effective_mask[None, :, :], y_sim, jnp.nan)


def _raise_if_log_link_mean_overflow(
    linear_predictors: jnp.ndarray,
    *,
    manifest_dist: str,
    manifest_dists: list[str] | None,
    manifest_links: list[str] | None,
    manifest_names: list[str] | None,
) -> None:
    """Fail fast when a log-link predictive mean would overflow before sampling."""
    _dists, links = resolve_manifest_families_and_links(
        manifest_dist,
        int(linear_predictors.shape[2]),
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
    )
    log_link_mask = np.asarray([link == LinkFunction.LOG for link in links], dtype=bool)
    if not bool(log_link_mask.any()):
        return

    linear_np = np.asarray(linear_predictors)
    overflow_threshold = float(np.log(np.finfo(linear_np.dtype).max))
    bad_mask = np.zeros_like(linear_np, dtype=bool)
    bad_mask[..., log_link_mask] = (
        ~np.isfinite(linear_np[..., log_link_mask])
    ) | (linear_np[..., log_link_mask] > overflow_threshold)
    if not bool(bad_mask.any()):
        return

    manifest_mask = bad_mask.any(axis=(0, 1))
    manifest_indices = tuple(int(idx) for idx in np.flatnonzero(manifest_mask))
    names = manifest_names or [f"var_{idx}" for idx in range(linear_np.shape[2])]
    bad_manifest_names = tuple(names[idx] for idx in manifest_indices if idx < len(names))
    draw_mask = bad_mask.reshape(bad_mask.shape[0], -1).any(axis=1)
    failing_draw_indices = tuple(int(idx) for idx in np.flatnonzero(draw_mask))
    time_mask = bad_mask.any(axis=(0, 2))
    first_bad_time_index = int(np.flatnonzero(time_mask)[0])
    max_linear_predictor = float(np.nanmax(linear_np[..., log_link_mask]))
    raise PredictiveObservationMeanOverflow(
        bad_manifest_names=bad_manifest_names,
        manifest_indices=manifest_indices,
        failing_draw_indices=failing_draw_indices,
        first_bad_time_index=first_bad_time_index,
        max_linear_predictor=max_linear_predictor,
        overflow_threshold=overflow_threshold,
    )


def _sample_observations_for_draw(
    linear_predictors: jnp.ndarray,
    rng_key: jax.Array,
    *,
    manifest_dist: str,
    manifest_dists: list[str] | None,
    manifest_links: list[str] | None,
    point_sampler,
    interval_summary_sampler,
    observation_operator,
    observation_mask: jnp.ndarray | None,
    extra_params: dict | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample one observation trajectory from precomputed linear predictors."""
    _key_latent, key_point, key_interval_summary = random.split(rng_key, 3)
    point_samples = point_sampler.sample_point_trajectory(key_point, linear_predictors)

    if not observation_operator.requires_interval_summary_handling:
        effective_mask = _resolve_effective_observation_mask(
            point_samples.shape,
            None,
            observation_mask,
        )
        return _apply_observation_mask(point_samples, None, observation_mask), effective_mask

    n_manifest = linear_predictors.shape[1]
    response_kernel = _build_response_kernel(
        manifest_dist,
        manifest_dists,
        manifest_links,
        n_manifest,
        extra_params,
    )
    responses = jax.vmap(response_kernel.response_fn)(linear_predictors)
    expected_means, semantic_mask = observation_operator.project_response_trajectory(responses)

    interval_summary_indices = list(observation_operator.interval_summary_indices)
    interval_summary_idx = jnp.asarray(interval_summary_indices, dtype=jnp.int32)
    assert interval_summary_sampler is not None
    sampled_interval_summary = interval_summary_sampler.sample_mean_trajectory(
        key_interval_summary,
        expected_means[:, interval_summary_idx],
    )
    point_samples = jax.vmap(lambda y_t, sampled_t: y_t.at[interval_summary_idx].set(sampled_t))(
        point_samples,
        sampled_interval_summary,
    )

    effective_mask = _resolve_effective_observation_mask(
        point_samples.shape,
        semantic_mask,
        observation_mask,
    )
    return _apply_observation_mask(point_samples, semantic_mask, observation_mask), effective_mask


def simulate_predictive_observations(
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    diffusion_dist: DistributionFamily | str = "gaussian",
    diffusion_dists: list[DistributionFamily | str] | None = None,
    manifest_dist: str = "gaussian",
    manifest_dists: list[str] | None = None,
    manifest_links: list[str] | None = None,
    manifest_level_counts: list[int] | None = None,
    observation_support: ObservationSupportRuntime | None = None,
    observation_mask: jnp.ndarray | None = None,
    n_subsample: int = 50,
    rng_seed: int = 42,
    manifest_names: list[str] | None = None,
    return_mask: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-simulate observations from predictive draws."""
    drift_draws = samples["drift"]
    diffusion_draws = samples["diffusion"]
    lambda_mat = samples["lambda"]
    manifest_cov = samples["manifest_cov"]
    t0_means = samples["t0_means"]
    t0_cov = samples["t0_cov"]

    cint_draws = samples.get("cint")
    manifest_means_draws = samples.get("manifest_means")

    n_draws_total = drift_draws.shape[0]
    n_use = min(n_subsample, n_draws_total)
    indices = jnp.linspace(0, n_draws_total - 1, n_use).astype(int)

    drift_sub = drift_draws[indices]
    diffusion_sub = diffusion_draws[indices]
    t0_means_sub = t0_means[indices]
    cint_sub = cint_draws[indices] if cint_draws is not None else None

    lambda_sub = _broadcast_draw_param(lambda_mat, n_use, indices)
    manifest_cov_sub = _broadcast_draw_param(manifest_cov, n_use, indices)
    t0_cov_sub = _broadcast_draw_param(t0_cov, n_use, indices)

    n_manifest = lambda_sub.shape[1]
    resolved_manifest_names = manifest_names or [f"var_{idx}" for idx in range(n_manifest)]
    if len(resolved_manifest_names) != n_manifest:
        raise ValueError(
            f"manifest_names must have length {n_manifest}, got {len(resolved_manifest_names)}"
        )

    if manifest_means_draws is not None:
        manifest_means_sub = _broadcast_draw_param(manifest_means_draws, n_use, indices)
    else:
        manifest_means_sub = jnp.zeros((n_use, n_manifest))

    ordered_cutpoints_sub = _broadcast_draw_param(
        samples.get("obs_ordered_cutpoints"), n_use, indices
    )
    cat_intercepts_sub = _broadcast_draw_param(samples.get("obs_cat_intercepts"), n_use, indices)
    cat_slopes_sub = _broadcast_draw_param(samples.get("obs_cat_slopes"), n_use, indices)
    obs_df_sub = _broadcast_draw_param(samples.get("obs_df"), n_use, indices)
    proc_df_sub = _broadcast_draw_param(samples.get("proc_df"), n_use, indices)
    obs_shape_sub = _broadcast_draw_param(samples.get("obs_shape"), n_use, indices)
    obs_r_sub = _broadcast_draw_param(samples.get("obs_r"), n_use, indices)
    obs_conc_sub = _broadcast_draw_param(samples.get("obs_concentration"), n_use, indices)
    level_counts = (
        jnp.asarray(manifest_level_counts, dtype=jnp.int32)
        if manifest_level_counts is not None
        else None
    )

    n_timepoints = int(times.shape[0])
    transition_dt_array = jnp.maximum(jnp.diff(times), MIN_DT)
    transition_semantics = compile_transition_semantics(
        diffusion_dists or diffusion_dist, drift_sub.shape[-1]
    )

    rng = jax.random.PRNGKey(rng_seed)
    draw_keys = jax.random.split(rng, n_use)

    resolved_dists, _resolved_links = resolve_manifest_families_and_links(
        manifest_dist,
        n_manifest,
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
    )
    observation_operator = compile_observation_operator(observation_support)

    if level_counts is None and any_family_needs_level_metadata([dist.value for dist in resolved_dists]):
        raise ValueError(
            "manifest_level_counts is required for ordered_logistic/categorical PPC simulation"
        )

    observation_mask_array = None
    if observation_mask is not None:
        observation_mask_array = jnp.asarray(observation_mask, dtype=bool)
        if observation_mask_array.shape != (n_timepoints, n_manifest):
            raise ValueError(
                "observation_mask must have shape (T, n_manifest) matching the predictive grid"
            )

    t0_chol_sub = vmap(
        lambda cov: stable_cholesky(cov, min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE)
    )(t0_cov_sub)

    if observation_operator.requires_interval_summary_handling:
        support = observation_operator.observation_support
        assert support is not None
        if support.anchor_times.shape != times.shape or not bool(
            jnp.allclose(jnp.asarray(support.anchor_times), times)
        ):
            raise ValueError("observation_support is not aligned to the predictive time grid")

    def _draw_extra_params(i: int) -> dict[str, jnp.ndarray | float]:
        extra_params: dict[str, jnp.ndarray | float] = {}
        if obs_df_sub is not None:
            extra_params["obs_df"] = obs_df_sub[i]
        if obs_shape_sub is not None:
            extra_params["obs_shape"] = obs_shape_sub[i]
        if obs_r_sub is not None:
            extra_params["obs_r"] = obs_r_sub[i]
        if obs_conc_sub is not None:
            extra_params["obs_concentration"] = obs_conc_sub[i]
        if level_counts is not None:
            extra_params["obs_level_counts"] = level_counts
        if ordered_cutpoints_sub is not None:
            extra_params["obs_ordered_cutpoints"] = ordered_cutpoints_sub[i]
        if cat_intercepts_sub is not None:
            extra_params["obs_cat_intercepts"] = cat_intercepts_sub[i]
        if cat_slopes_sub is not None:
            extra_params["obs_cat_slopes"] = cat_slopes_sub[i]
        return extra_params

    def _simulate_linear_predictors_for_draw(i):
        key_latent, _key_point, _key_interval_summary = random.split(draw_keys[i], 3)
        latent_trajectory = _simulate_latent_trajectory(
            drift=drift_sub[i],
            diffusion_chol=diffusion_sub[i],
            cint=cint_sub[i] if cint_sub is not None else None,
            t0_mean=t0_means_sub[i],
            t0_chol=t0_chol_sub[i],
            transition_dt_array=transition_dt_array,
            n_timepoints=n_timepoints,
            rng_key=key_latent,
            transition_semantics=transition_semantics,
            proc_df=proc_df_sub[i] if proc_df_sub is not None else 5.0,
        )
        return _linear_predictors_from_latent_trajectory(
            latent_trajectory,
            lambda_sub[i],
            manifest_means_sub[i],
        )

    linear_predictors_sub = vmap(_simulate_linear_predictors_for_draw)(jnp.arange(n_use))
    _raise_if_log_link_mean_overflow(
        linear_predictors_sub,
        manifest_dist=manifest_dist,
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
        manifest_names=resolved_manifest_names,
    )

    def sim_one(i):
        extra_params = _draw_extra_params(i)
        point_sampler = build_predictive_observation_sampler(
            manifest_dist=manifest_dist,
            manifest_cov=manifest_cov_sub[i],
            manifest_dists=manifest_dists,
            manifest_links=manifest_links,
            extra_params=extra_params,
        )
        interval_summary_sampler = None
        if observation_operator.requires_interval_summary_handling:
            interval_summary_indices = list(observation_operator.interval_summary_indices)
            interval_summary_idx = jnp.asarray(interval_summary_indices, dtype=jnp.int32)
            interval_summary_sampler = build_predictive_observation_sampler(
                manifest_dist=manifest_dist,
                manifest_cov=manifest_cov_sub[i][
                    jnp.ix_(interval_summary_idx, interval_summary_idx)
                ],
                manifest_dists=[
                    point_sampler.manifest_dists[idx] for idx in interval_summary_indices
                ],
                manifest_links=(
                    [manifest_links[idx] for idx in interval_summary_indices]
                    if manifest_links is not None
                    else None
                ),
                extra_params=_slice_extra_params_for_indices(
                    extra_params, interval_summary_indices
                ),
            )
        return _sample_observations_for_draw(
            linear_predictors=linear_predictors_sub[i],
            rng_key=draw_keys[i],
            manifest_dist=manifest_dist,
            manifest_dists=manifest_dists,
            manifest_links=manifest_links,
            point_sampler=point_sampler,
            interval_summary_sampler=interval_summary_sampler,
            observation_operator=observation_operator,
            observation_mask=observation_mask_array,
            extra_params=extra_params,
        )

    y_sim, y_mask = vmap(sim_one)(jnp.arange(n_use))

    if return_mask:
        return y_sim, y_mask
    return y_sim
