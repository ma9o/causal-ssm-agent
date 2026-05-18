"""Runtime dispatch for observation-family behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from nof1_causal_lab.artifacts.model_spec import DistributionFamily
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter

from .base import NUMERICAL_EPSILON
from .emissions import build_composite_mean_sample_fn
from .observation_families import (
    FAMILY_REGISTRY,
    POSTERIOR_PREDICTIVE_SWITCH_BRANCHES,
    get_posterior_predictive_switch_index,
    resolve_manifest_families_and_links,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def get_emission_score_weight_fn(manifest_dist, extra_params=None, *, link=None):
    """Return analytical (score, neg_hess_diag) w.r.t. linear predictor eta."""
    extra_params = extra_params or {}
    dist = DistributionFamily(manifest_dist)
    family_spec = FAMILY_REGISTRY.get(dist)
    if family_spec is None:
        return None
    link_key = str(link) if link else "default"
    factory = family_spec.score_weight_fns.get(link_key) or family_spec.score_weight_fns.get(
        "default"
    )
    if factory is None:
        return None
    return factory(extra_params)


def get_emission_fn(manifest_dist, extra_params=None, *, link=None):
    """Resolve the emission log-probability function for a family/link pair."""
    extra_params = extra_params or {}
    try:
        dist = DistributionFamily(manifest_dist)
    except ValueError as exc:
        raise ValueError(
            f"No emission function for manifest_dist='{manifest_dist}'. "
            "Supported: gaussian, student_t, poisson, gamma, bernoulli, "
            "negative_binomial, beta, ordered_logistic, categorical."
        ) from exc

    family_spec = FAMILY_REGISTRY.get(dist)
    if family_spec is None:
        raise ValueError(
            f"No emission function for manifest_dist='{manifest_dist}'. "
            "Supported: gaussian, student_t, poisson, gamma, bernoulli, "
            "negative_binomial, beta, ordered_logistic, categorical."
        )

    link_key = str(link) if link else "default"
    factory = family_spec.emission_fns.get(link_key) or family_spec.emission_fns.get("default")
    if factory is None:
        raise ValueError(
            f"No emission function for manifest_dist='{manifest_dist}', link='{link}'."
        )
    return factory(extra_params)


@dataclass(frozen=True)
class PredictiveObservationSampler:
    """Compiled predictive sampler shared by posterior/prior predictive paths."""

    sample_point_trajectory: Callable[[jax.Array, jnp.ndarray], jnp.ndarray]
    sample_mean_trajectory: Callable[[jax.Array, jnp.ndarray], jnp.ndarray]
    all_gaussian: bool
    manifest_dists: tuple[str, ...]


def build_predictive_observation_sampler(
    manifest_dists,
    manifest_cov: jnp.ndarray,
    *,
    manifest_links=None,
    extra_params: dict | None = None,
) -> PredictiveObservationSampler:
    """Compile predictive samplers for point observations and mean-space summaries."""
    dists, links = resolve_manifest_families_and_links(
        manifest_dists,
        manifest_links=manifest_links,
    )
    n_manifest = len(dists)
    all_gaussian = all(dist == DistributionFamily.GAUSSIAN for dist in dists)
    manifest_dist_values = tuple(dist.value for dist in dists)
    try:
        mean_sample_fn = build_composite_mean_sample_fn(manifest_dist_values, extra_params)
    except ValueError as exc:
        mean_sample_fn = None
        mean_sampler_error = exc
    else:
        mean_sampler_error = None

    def _sample_mean_vector(key, mean_t):
        if mean_sample_fn is None:
            raise ValueError(
                f"Mean-parameter sampler is not defined for manifest_dists={manifest_dist_values}."
            ) from mean_sampler_error
        return mean_sample_fn(key, mean_t, manifest_cov)

    def _sample_mean_trajectory(key, mean_trajectory):
        mean_keys = jax.random.split(key, mean_trajectory.shape[0])
        return jax.vmap(_sample_mean_vector)(mean_keys, mean_trajectory)

    if all_gaussian:
        manifest_cov_adj = symmetrize_with_jitter(manifest_cov)
        manifest_chol = jnp.linalg.cholesky(manifest_cov_adj)

        def _sample_point_vector(key, linear_predictor):
            return linear_predictor + manifest_chol @ jax.random.normal(key, linear_predictor.shape)

        def _sample_point_trajectory(key, linear_predictors):
            point_keys = jax.random.split(key, linear_predictors.shape[0])
            return jax.vmap(_sample_point_vector)(point_keys, linear_predictors)

        return PredictiveObservationSampler(
            sample_point_trajectory=_sample_point_trajectory,
            sample_mean_trajectory=_sample_mean_trajectory,
            all_gaussian=True,
            manifest_dists=manifest_dist_values,
        )

    dist_indices = jnp.asarray(
        [
            get_posterior_predictive_switch_index(dist, link=link)
            for dist, link in zip(dists, links, strict=False)
        ],
        dtype=jnp.int64,
    )
    manifest_std = jnp.sqrt(jnp.maximum(jnp.diag(manifest_cov), NUMERICAL_EPSILON))
    params = extra_params or {}
    level_counts = params.get("obs_level_counts")
    if level_counts is None:
        level_counts = jnp.ones((n_manifest,), dtype=jnp.int64)
    else:
        level_counts = jnp.asarray(level_counts, dtype=jnp.int64)
    ordered_cutpoints = params.get("obs_ordered_cutpoints")
    if ordered_cutpoints is None:
        ordered_cutpoints = jnp.zeros((n_manifest, 1), dtype=manifest_cov.dtype)
    cat_intercepts = params.get("obs_cat_intercepts")
    if cat_intercepts is None:
        cat_intercepts = jnp.zeros((n_manifest, 1), dtype=manifest_cov.dtype)
    cat_slopes = params.get("obs_cat_slopes")
    if cat_slopes is None:
        cat_slopes = jnp.zeros((n_manifest, 1), dtype=manifest_cov.dtype)
    obs_df = jnp.asarray(params.get("obs_df", 5.0), dtype=manifest_cov.dtype)
    obs_shape = jnp.asarray(params.get("obs_shape", 2.0), dtype=manifest_cov.dtype)
    obs_r = jnp.asarray(params.get("obs_r", 5.0), dtype=manifest_cov.dtype)
    obs_concentration = jnp.asarray(
        params.get("obs_concentration", 10.0),
        dtype=manifest_cov.dtype,
    )

    def _sample_channel(
        loc_j,
        key,
        dist_idx,
        std_j,
        df,
        shape_p,
        r_p,
        phi_p,
        level_count,
        cutpoints,
        cat_intercepts_j,
        cat_slopes_j,
    ):
        return jax.lax.switch(
            dist_idx,
            POSTERIOR_PREDICTIVE_SWITCH_BRANCHES,
            loc_j,
            key,
            std_j,
            df,
            shape_p,
            r_p,
            phi_p,
            level_count,
            cutpoints,
            cat_intercepts_j,
            cat_slopes_j,
        )

    def _sample_point_vector(key, linear_predictor):
        channel_keys = jax.random.split(key, n_manifest)
        return jax.vmap(_sample_channel)(
            linear_predictor,
            channel_keys,
            dist_indices,
            manifest_std,
            jnp.full((n_manifest,), obs_df),
            jnp.full((n_manifest,), obs_shape),
            jnp.full((n_manifest,), obs_r),
            jnp.full((n_manifest,), obs_concentration),
            level_counts,
            ordered_cutpoints,
            cat_intercepts,
            cat_slopes,
        )

    def _sample_point_trajectory(key, linear_predictors):
        point_keys = jax.random.split(key, linear_predictors.shape[0])
        return jax.vmap(_sample_point_vector)(point_keys, linear_predictors)

    return PredictiveObservationSampler(
        sample_point_trajectory=_sample_point_trajectory,
        sample_mean_trajectory=_sample_mean_trajectory,
        all_gaussian=False,
        manifest_dists=manifest_dist_values,
    )
