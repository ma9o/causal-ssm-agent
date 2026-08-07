"""Shared predictive observation simulation for prior and posterior workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap

from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.execution.observation_families import (
    any_family_needs_level_metadata,
    resolve_manifest_families_and_links,
)
from nof1_causal_lab.models.ssm.inference.targets.kernels import (
    CompiledObservationModel,
    compile_observation_model,
)
from nof1_causal_lab.models.ssm.inference.targets.trajectory_observations import (
    compile_observation_operator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

logger = logging.getLogger(__name__)
_DEFAULT_PREDICTIVE_KEY = random.PRNGKey(42)


class PredictiveObservationMeanOverflow(RuntimeError):
    """Raised when a log-link predictive mean exceeds finite numeric range."""

    def __init__(
        self,
        *,
        bad_manifest_names: tuple[str, ...],
        manifest_indices: tuple[int, ...],
        failing_draw_indices: tuple[int, ...],
        n_draws: int,
        first_bad_time_index: int,
        max_linear_predictor: float,
        overflow_threshold: float,
        n_nonfinite: int = 0,
    ) -> None:
        self.bad_manifest_names = bad_manifest_names
        self.manifest_indices = manifest_indices
        self.failing_draw_indices = failing_draw_indices
        self.n_draws = n_draws
        self.first_bad_time_index = first_bad_time_index
        self.max_linear_predictor = max_linear_predictor
        self.overflow_threshold = overflow_threshold
        self.n_nonfinite = n_nonfinite
        manifest_summary = ", ".join(bad_manifest_names) if bad_manifest_names else "unknown"
        if n_nonfinite:
            cause = (
                f"linear predictor contains {n_nonfinite} non-finite (NaN/Inf) values for "
                f"{manifest_summary} — the latent simulation diverged under these priors "
                "(rein in feedback edge gains, diffusion, or persistence), "
                f"max finite eta={max_linear_predictor:.2f}"
            )
        else:
            cause = (
                f"linear predictor exceeded the finite exp range for {manifest_summary} "
                f"(max eta={max_linear_predictor:.2f}, threshold={overflow_threshold:.2f}"
            )
        super().__init__(
            "Predictive log-link mean overflow before observation sampling: "
            f"{cause}, first bad time index={first_bad_time_index}; "
            f"{len(failing_draw_indices)} of {n_draws} prior draws affected — "
            "a small fraction means heavy tails (tighten sigma/upper bounds); "
            "most draws means the central mass is wrong (lower the loading/edge-gain "
            "or intercept location feeding this log link)."
        )


def _broadcast_draw_param(
    value: jnp.ndarray,
    n_use: int,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    if value.ndim == 0:
        return jnp.broadcast_to(value, (n_use,))
    if value.shape[0] == n_use:
        return value
    if value.shape[0] >= int(indices[-1]) + 1:
        return value[indices]
    return jnp.broadcast_to(value, (n_use, *value.shape))


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
    manifest_dists: Sequence[DistributionFamily | str],
    manifest_links: Sequence[LinkFunction | str | None] | None,
    manifest_names: list[str] | None,
) -> None:
    """Fail fast when a log-link predictive mean would overflow before sampling."""
    _dists, links = resolve_manifest_families_and_links(
        list(manifest_dists),
        manifest_links=list(manifest_links) if manifest_links is not None else None,
    )
    log_link_mask = np.asarray([link == LinkFunction.LOG for link in links], dtype=bool)
    if not bool(log_link_mask.any()):
        return

    linear_np = np.asarray(linear_predictors)
    overflow_threshold = float(np.log(np.finfo(linear_np.dtype).max))
    bad_mask = np.zeros_like(linear_np, dtype=bool)
    bad_mask[..., log_link_mask] = (~np.isfinite(linear_np[..., log_link_mask])) | (
        linear_np[..., log_link_mask] > overflow_threshold
    )
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
        n_draws=int(linear_np.shape[0]),
        first_bad_time_index=first_bad_time_index,
        max_linear_predictor=max_linear_predictor,
        overflow_threshold=overflow_threshold,
        n_nonfinite=int((~np.isfinite(linear_np[..., log_link_mask])).sum()),
    )


def _sample_observations_for_draw(
    linear_predictors: jnp.ndarray,
    rng_key: jax.Array,
    *,
    observation_model: CompiledObservationModel,
    observation_mask: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample one observation trajectory from precomputed linear predictors."""
    _key_latent, key_point, key_interval_summary = random.split(rng_key, 3)
    point_samples = observation_model.point_sampler.sample_point_trajectory(
        key_point,
        linear_predictors,
    )
    responses = jax.vmap(observation_model.kernel.response_fn)(linear_predictors)
    observation_operator = observation_model.observation_operator

    if not observation_operator.requires_interval_summary_handling:
        effective_mask = _resolve_effective_observation_mask(
            point_samples.shape,
            None,
            observation_mask,
        )
        return (
            _apply_observation_mask(point_samples, None, observation_mask),
            effective_mask,
            _apply_observation_mask(responses, None, observation_mask),
        )

    expected_means, semantic_mask = observation_operator.project_response_trajectory(responses)

    interval_summary_indices = list(observation_operator.interval_summary_indices)
    interval_summary_idx = jnp.asarray(interval_summary_indices, dtype=jnp.int32)
    assert observation_model.interval_summary_sampler is not None
    sampled_interval_summary = observation_model.interval_summary_sampler.sample_mean_trajectory(
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
    return (
        _apply_observation_mask(point_samples, semantic_mask, observation_mask),
        effective_mask,
        _apply_observation_mask(expected_means, semantic_mask, observation_mask),
    )


def sample_predictive_observations_from_linear_predictors(
    linear_predictors: jnp.ndarray,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    *,
    rng_key: jax.Array = _DEFAULT_PREDICTIVE_KEY,
    manifest_dists: Sequence[DistributionFamily | str] | None = None,
    manifest_links: Sequence[LinkFunction | str | None] | None = None,
    manifest_level_counts: list[int] | None = None,
    observation_support: ObservationSupportRuntime | None = None,
    observation_mask: jnp.ndarray | None = None,
    n_subsample: int = 50,
    manifest_names: list[str] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample observations from precomputed observation linear predictors."""
    linear_predictors = jnp.asarray(linear_predictors)
    if linear_predictors.ndim != 3:
        raise ValueError(
            "linear_predictors must have shape (n_draws, n_timepoints, n_manifest), "
            f"got {linear_predictors.shape}."
        )

    n_draws_total = linear_predictors.shape[0]
    n_use = min(n_subsample, n_draws_total)
    indices = jnp.linspace(0, n_draws_total - 1, n_use).astype(int)

    linear_predictors_sub = linear_predictors[indices]
    manifest_cov_sub = _broadcast_draw_param(samples["manifest_cov"], n_use, indices)

    n_timepoints = int(times.shape[0])
    n_manifest = linear_predictors_sub.shape[2]
    resolved_manifest_names = manifest_names or [f"var_{idx}" for idx in range(n_manifest)]
    if len(resolved_manifest_names) != n_manifest:
        raise ValueError(
            f"manifest_names must have length {n_manifest}, got {len(resolved_manifest_names)}"
        )
    resolved_manifest_dists = (
        list(manifest_dists) if manifest_dists is not None else ["gaussian"] * n_manifest
    )
    resolved_manifest_links = list(manifest_links) if manifest_links is not None else None

    ordered_cutpoints_draws = samples.get("obs_ordered_cutpoints")
    ordered_cutpoints_sub = (
        _broadcast_draw_param(ordered_cutpoints_draws, n_use, indices)
        if ordered_cutpoints_draws is not None
        else None
    )
    cat_intercepts_draws = samples.get("obs_cat_intercepts")
    cat_intercepts_sub = (
        _broadcast_draw_param(cat_intercepts_draws, n_use, indices)
        if cat_intercepts_draws is not None
        else None
    )
    cat_slopes_draws = samples.get("obs_cat_slopes")
    cat_slopes_sub = (
        _broadcast_draw_param(cat_slopes_draws, n_use, indices)
        if cat_slopes_draws is not None
        else None
    )
    obs_df_draws = samples.get("obs_df")
    obs_df_sub = (
        _broadcast_draw_param(obs_df_draws, n_use, indices) if obs_df_draws is not None else None
    )
    obs_shape_draws = samples.get("obs_shape")
    obs_shape_sub = (
        _broadcast_draw_param(obs_shape_draws, n_use, indices)
        if obs_shape_draws is not None
        else None
    )
    obs_r_draws = samples.get("obs_r")
    obs_r_sub = (
        _broadcast_draw_param(obs_r_draws, n_use, indices) if obs_r_draws is not None else None
    )
    obs_concentration_draws = samples.get("obs_concentration")
    obs_conc_sub = (
        _broadcast_draw_param(obs_concentration_draws, n_use, indices)
        if obs_concentration_draws is not None
        else None
    )
    level_counts = (
        jnp.asarray(manifest_level_counts, dtype=jnp.int32)
        if manifest_level_counts is not None
        else None
    )

    draw_keys = jax.random.split(rng_key, n_use)

    resolved_dists, _resolved_links = resolve_manifest_families_and_links(
        resolved_manifest_dists,
        manifest_links=resolved_manifest_links,
    )
    observation_operator = compile_observation_operator(observation_support)

    if level_counts is None and any_family_needs_level_metadata(resolved_dists):
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

    if observation_operator.requires_interval_summary_handling:
        support = observation_operator.observation_support
        assert support is not None
        if support.anchor_times.shape != times.shape or not bool(
            jnp.allclose(jnp.asarray(support.anchor_times), times)
        ):
            raise ValueError("observation_support is not aligned to the predictive time grid")

    _raise_if_log_link_mean_overflow(
        linear_predictors_sub,
        manifest_dists=resolved_manifest_dists,
        manifest_links=manifest_links,
        manifest_names=resolved_manifest_names,
    )

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

    def sim_one(i):
        extra_params = _draw_extra_params(i)
        observation_model = compile_observation_model(
            resolved_manifest_dists,
            manifest_cov=manifest_cov_sub[i],
            manifest_links=resolved_manifest_links,
            extra_params=extra_params,
            observation_support=observation_support,
        )
        return _sample_observations_for_draw(
            linear_predictors=linear_predictors_sub[i],
            rng_key=draw_keys[i],
            observation_model=observation_model,
            observation_mask=observation_mask_array,
        )

    y_sim, y_mask, expected = vmap(sim_one)(jnp.arange(n_use))
    return y_sim, y_mask, expected
