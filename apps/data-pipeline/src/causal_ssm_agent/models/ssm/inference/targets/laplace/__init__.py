"""Laplace-approximated likelihood backend for non-Gaussian SSMs.

Computes log p(y|theta) by combining an Iterated Extended Kalman Smoother
(IEKS) mode-finding inner loop with a Laplace approximation to the marginal
likelihood.  Three solver strategies are dispatched automatically:

- **Point IEKS**: block-tridiagonal O(T D^3) — used when every observation is
  a point measurement.
- **Support-aware IEKS**: profile-banded Cholesky — used when some observations
  are interval summaries (e.g. means/sums over windows).
- **Dense support**: full joint Hessian — fallback for very short series with
  interval summaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.ssm.discretization import discretize_system_with_inputs_batched
from causal_ssm_agent.models.ssm.inference.targets.kernels import compile_measurement_semantics
from causal_ssm_agent.models.ssm.inference.targets.linear_summary_augmentation import (
    build_linear_summary_augmented_system as _build_linear_summary_augmented_system,
)
from causal_ssm_agent.models.ssm.inference.targets.trajectory_observations import (
    get_summary_operator_codes,
    get_support_kind_codes,
)

from .point import (
    _dense_support_laplace_log_lik,
    _ieks_smooth,
    _linear_summary_augmented_ieks_laplace,
    _point_ieks_mode,
    _point_laplace_from_mode,
)
from .shared import (
    _block_banded_logdet,
    _build_ieks_system_from_prior,
    _build_linear_summary_accumulator_plan,
    _build_prior_tridiagonal_system,
    _compute_profile_lower_bandwidths,
    _factor_block_banded_cholesky,
    _factor_block_profile_cholesky,
    _infer_support_groups,
    _predictive_latent_init,
    _should_use_dense_support_laplace,
    _solve_block_banded_from_cholesky,
    _solve_block_profile_from_cholesky,
    _solve_block_tridiagonal,
    _tree_contains_tracer,
    block_profile_logdet_packed_cotangent,
)
from .support import (
    _assemble_support_aware_observation_system,
    _make_support_window_derivatives,
    _support_aware_ieks_laplace,
    _support_aware_ieks_mode,
    _support_aware_laplace_from_mode,
    _support_aware_step_halving_search,
)

if TYPE_CHECKING:
    from causal_ssm_agent.artifacts.model_spec import DistributionFamily, LinkFunction
    from causal_ssm_agent.models.ssm.inference.targets.base import (
        CTParams,
        InitialStateParams,
        MeasurementParams,
    )
    from causal_ssm_agent.models.ssm_observation_metadata import ObservationSupportRuntime


class LaplaceLikelihood:
    """Laplace-approximated likelihood backend.

    Computes log p(y|theta) via IEKS + Laplace approximation.
    Drop-in replacement for KalmanLikelihood / ParticleLikelihood.

    Accepts per-channel distribution and link lists to support heterogeneous
    observation models (e.g., channel 0 Gaussian, channel 1 Poisson).
    """

    # The support-aware Laplace path constructs runtime callables and custom-VJP
    # closures that are not remat-safe under large traced outer evaluations.
    checkpoint_loglik = False

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        manifest_dists: list[DistributionFamily],
        manifest_links: list[LinkFunction],
        n_ieks_iters: int = 5,
        observation_support: ObservationSupportRuntime | None = None,
    ):
        self.n_latent = n_latent
        self.n_manifest = n_manifest
        self.manifest_dists = manifest_dists
        self.manifest_links = manifest_links
        self.n_ieks_iters = n_ieks_iters
        self.observation_support = observation_support
        self._point_mode_cache: jnp.ndarray | None = None
        self._support_mode_cache: jnp.ndarray | None = None
        self._linear_summary_mode_cache: jnp.ndarray | None = None
        self._support_window_derivatives = None
        self._support_window_derivatives_signature: tuple[Any, ...] | None = None
        self._linear_summary_plan = _build_linear_summary_accumulator_plan(
            observation_support,
            manifest_dists,
            manifest_links,
        )
        if observation_support is not None:
            self._support_kind_codes = get_support_kind_codes(observation_support)
            self._summary_operator_codes = get_summary_operator_codes(observation_support)
        else:
            self._support_kind_codes = jnp.zeros((n_manifest,), dtype=jnp.int64)
            self._summary_operator_codes = jnp.zeros((n_manifest,), dtype=jnp.int64)
        if (
            observation_support is not None
            and observation_support.requires_interval_summary_handling
        ):
            (
                self._support_window_batches,
                self._support_bandwidth,
                support_row_upper_bandwidths,
            ) = _infer_support_groups(observation_support)
            prior_row_upper_bandwidths = np.zeros(
                (len(observation_support.anchor_times),),
                dtype=np.int64,
            )
            if len(prior_row_upper_bandwidths) > 1:
                prior_row_upper_bandwidths[:-1] = 1
            full_row_upper_bandwidths = np.maximum(
                np.asarray(support_row_upper_bandwidths, dtype=np.int64),
                prior_row_upper_bandwidths,
            )
            self._support_row_upper_bandwidths = jnp.asarray(
                full_row_upper_bandwidths,
                dtype=jnp.int32,
            )
            self._support_row_lower_bandwidths = jnp.asarray(
                _compute_profile_lower_bandwidths(full_row_upper_bandwidths),
                dtype=jnp.int32,
            )
        else:
            self._support_window_batches = ()
            self._support_bandwidth = 1 if n_latent > 0 else 0
            self._support_row_upper_bandwidths = jnp.zeros((0,), dtype=jnp.int32)
            self._support_row_lower_bandwidths = jnp.zeros((0,), dtype=jnp.int32)

    def _build_support_window_derivatives(self, measurement_semantics) -> tuple[Any, ...]:
        return tuple(
            _make_support_window_derivatives(
                max_state_len=batch.max_state_len,
                n_latent=self.n_latent,
                n_manifest=self.n_manifest,
                summary_operator_codes=self._summary_operator_codes,
                obs_kernel=measurement_semantics.obs_kernel,
                mean_log_prob_fn=measurement_semantics.mean_log_prob_fn,
            )
            for batch in self._support_window_batches
        )

    def _get_support_window_derivatives(
        self,
        measurement_semantics,
        extra_params: dict | None,
        *,
        allow_cache: bool,
    ):
        if not allow_cache or extra_params is not None:
            return self._build_support_window_derivatives(measurement_semantics)

        signature = (
            measurement_semantics.manifest_dists,
            measurement_semantics.manifest_links,
            tuple(batch.max_state_len for batch in self._support_window_batches),
            self.n_latent,
            self.n_manifest,
        )
        if (
            self._support_window_derivatives is None
            or self._support_window_derivatives_signature != signature
        ):
            self._support_window_derivatives = self._build_support_window_derivatives(
                measurement_semantics
            )
            self._support_window_derivatives_signature = signature
        return self._support_window_derivatives

    def _compute_log_likelihood_impl(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        *,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
        latent_mode_init: jnp.ndarray | None = None,
        transition_inputs: jnp.ndarray | None = None,
        include_aux: bool,
        allow_stateful_cache: bool,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray] | None]:
        """Shared Laplace likelihood implementation with explicit cache control."""
        n = self.n_latent

        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)
        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        with jax.named_scope("map/compile_measurement_semantics"):
            measurement_semantics = compile_measurement_semantics(
                self.manifest_dists,
                manifest_cov=measurement_params.manifest_cov,
                extra_params=extra_params,
                manifest_links=self.manifest_links,
                observation_support=self.observation_support,
            )
        obs_kernel = measurement_semantics.obs_kernel

        def _discretize_base_system() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            with jax.named_scope("map/discretize_system"):
                Ad, Qd, cd = discretize_system_with_inputs_batched(
                    ct_params.drift,
                    ct_params.diffusion_cov,
                    ct_params.cint,
                    ct_params.input_effect,
                    transition_inputs,
                    time_intervals,
                )
            if cd is None:
                cd = jnp.zeros((len(time_intervals), n))
            else:
                cd = jnp.asarray(cd)
                if cd.ndim == 1:
                    cd = cd[:, None]
            return Ad, Qd, cd

        if (
            self.observation_support is not None
            and self.observation_support.requires_interval_summary_handling
        ):
            cache_inputs = (
                ct_params,
                measurement_params,
                initial_state,
                observations,
                time_intervals,
                obs_mask,
                extra_params,
            )
            if self._linear_summary_plan is not None:

                def _build_linear_summary_measurement_objects(
                    manifest_cov: jnp.ndarray,
                    runtime_extra_params: dict | None,
                ):
                    return compile_measurement_semantics(
                        self.manifest_dists,
                        manifest_cov=manifest_cov,
                        extra_params=runtime_extra_params,
                        manifest_links=self.manifest_links,
                        observation_support=self.observation_support,
                    )

                can_reuse_linear_summary_mode = allow_stateful_cache and not _tree_contains_tracer(
                    cache_inputs
                )
                linear_summary_dim = self.n_latent + self._linear_summary_plan.n_accumulators
                linear_summary_mode_init = latent_mode_init
                if linear_summary_mode_init is not None and linear_summary_mode_init.shape != (
                    clean_obs.shape[0],
                    linear_summary_dim,
                ):
                    raise ValueError(
                        "Linear interval-summary warm start shape does not match the "
                        f"augmented latent dimension: expected {(clean_obs.shape[0], linear_summary_dim)}, "
                        f"received {tuple(linear_summary_mode_init.shape)}."
                    )
                if (
                    linear_summary_mode_init is None
                    and can_reuse_linear_summary_mode
                    and self._linear_summary_mode_cache is not None
                    and self._linear_summary_mode_cache.shape
                    == (clean_obs.shape[0], linear_summary_dim)
                ):
                    linear_summary_mode_init = self._linear_summary_mode_cache
                with jax.named_scope("map/linear_summary_augmented_backend"):
                    z_mode, log_lik, inner_eval_aux = _linear_summary_augmented_ieks_laplace(
                        clean_obs,
                        obs_mask,
                        time_intervals,
                        ct_params.drift,
                        ct_params.diffusion_cov,
                        ct_params.cint,
                        measurement_params.lambda_mat,
                        measurement_params.manifest_means,
                        measurement_params.manifest_cov,
                        initial_state.mean,
                        initial_state.cov,
                        obs_kernel,
                        self._linear_summary_plan,
                        self._support_kind_codes,
                        self.n_ieks_iters,
                        z_init=linear_summary_mode_init,
                        build_measurement_objects=_build_linear_summary_measurement_objects,
                        extra_params=extra_params,
                    )
                    if can_reuse_linear_summary_mode:
                        self._linear_summary_mode_cache = jax.device_get(z_mode)
                    return log_lik, inner_eval_aux if include_aux else None

            def _build_support_measurement_objects(
                manifest_cov: jnp.ndarray,
                runtime_extra_params: dict | None,
            ):
                runtime_measurement_semantics = compile_measurement_semantics(
                    self.manifest_dists,
                    manifest_cov=manifest_cov,
                    extra_params=runtime_extra_params,
                    manifest_links=self.manifest_links,
                    observation_support=self.observation_support,
                )
                allow_runtime_cache = allow_stateful_cache and not _tree_contains_tracer(
                    (manifest_cov, runtime_extra_params)
                )
                return runtime_measurement_semantics, self._get_support_window_derivatives(
                    runtime_measurement_semantics,
                    runtime_extra_params,
                    allow_cache=allow_runtime_cache,
                )

            Ad, Qd, cd = _discretize_base_system()
            can_reuse_support_mode = allow_stateful_cache and not _tree_contains_tracer(
                cache_inputs
            )
            can_cache_window_derivatives = allow_stateful_cache and not _tree_contains_tracer(
                (measurement_params.manifest_cov, extra_params)
            )
            support_mode_init = latent_mode_init
            if (
                support_mode_init is None
                and can_reuse_support_mode
                and self._support_mode_cache is not None
                and self._support_mode_cache.shape == (clean_obs.shape[0], self.n_latent)
            ):
                support_mode_init = self._support_mode_cache
            if _should_use_dense_support_laplace(
                n_time=clean_obs.shape[0],
                n_latent=self.n_latent,
            ):
                with jax.named_scope("map/dense_support_backend"):
                    log_lik, inner_eval_aux = _dense_support_laplace_log_lik(
                        clean_obs,
                        obs_mask,
                        Ad,
                        Qd,
                        cd,
                        measurement_params.lambda_mat,
                        measurement_params.manifest_means,
                        measurement_params.manifest_cov,
                        initial_state.mean,
                        initial_state.cov,
                        obs_kernel,
                        measurement_semantics.mean_log_prob_fn,
                        self.observation_support,
                        self.n_ieks_iters,
                    )
                return log_lik, inner_eval_aux if include_aux else None
            with jax.named_scope("map/support_aware_backend"):
                window_derivatives = self._get_support_window_derivatives(
                    measurement_semantics,
                    extra_params,
                    allow_cache=can_cache_window_derivatives,
                )
                log_lik, z_mode, inner_eval_aux = _support_aware_ieks_laplace(
                    clean_obs,
                    obs_mask,
                    Ad,
                    Qd,
                    cd,
                    measurement_params.lambda_mat,
                    measurement_params.manifest_means,
                    measurement_params.manifest_cov,
                    initial_state.mean,
                    initial_state.cov,
                    obs_kernel,
                    measurement_semantics.mean_log_prob_fn,
                    self.observation_support,
                    self._support_window_batches,
                    self._support_bandwidth,
                    self._support_row_upper_bandwidths,
                    self._support_row_lower_bandwidths,
                    window_derivatives=window_derivatives,
                    build_measurement_objects=_build_support_measurement_objects,
                    extra_params=extra_params,
                    n_ieks_iters=self.n_ieks_iters,
                    z_init=support_mode_init,
                )
                if can_reuse_support_mode:
                    self._support_mode_cache = jax.device_get(z_mode)
                return log_lik, inner_eval_aux if include_aux else None

        cache_inputs = (
            ct_params,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            obs_mask,
            extra_params,
        )
        can_reuse_point_mode = allow_stateful_cache and not _tree_contains_tracer(cache_inputs)
        point_mode_init = latent_mode_init
        if (
            point_mode_init is None
            and can_reuse_point_mode
            and self._point_mode_cache is not None
            and self._point_mode_cache.shape == (clean_obs.shape[0], self.n_latent)
        ):
            point_mode_init = self._point_mode_cache

        Ad, Qd, cd = _discretize_base_system()
        T_obs = clean_obs.shape[0]
        H_rows = jnp.broadcast_to(
            measurement_params.lambda_mat[None, :, :],
            (T_obs, *measurement_params.lambda_mat.shape),
        )
        d_rows = jnp.broadcast_to(
            measurement_params.manifest_means[None, :],
            (T_obs, *measurement_params.manifest_means.shape),
        )

        def _build_point_measurement_objects(
            manifest_cov: jnp.ndarray,
            runtime_extra_params: dict | None,
        ):
            return compile_measurement_semantics(
                self.manifest_dists,
                manifest_cov=manifest_cov,
                extra_params=runtime_extra_params,
                manifest_links=self.manifest_links,
                observation_support=self.observation_support,
            )

        with jax.named_scope("map/ieks_backend"):
            z_mode, log_lik, inner_eval_aux = _ieks_smooth(
                clean_obs,
                obs_mask,
                Ad,
                Qd,
                cd,
                H_rows,
                d_rows,
                measurement_params.manifest_cov,
                initial_state.mean,
                initial_state.cov,
                obs_kernel,
                n_ieks_iters=self.n_ieks_iters,
                z_init=point_mode_init,
                build_measurement_objects=_build_point_measurement_objects,
                extra_params=extra_params,
            )
            if can_reuse_point_mode:
                self._point_mode_cache = jax.device_get(z_mode)

        return log_lik, inner_eval_aux if include_aux else None

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
        latent_mode_init: jnp.ndarray | None = None,
        transition_inputs: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute Laplace-approximated log-likelihood.

        Returns:
            (T,) cumulative log-normalizing constants, matching LikelihoodBackend protocol.
        """
        log_lik, _aux = self._compute_log_likelihood_impl(
            ct_params,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            obs_mask=obs_mask,
            extra_params=extra_params,
            latent_mode_init=latent_mode_init,
            transition_inputs=transition_inputs,
            include_aux=False,
            allow_stateful_cache=False,
        )
        return log_lik

    def compute_log_likelihood_with_aux(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
        latent_mode_init: jnp.ndarray | None = None,
        transition_inputs: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute Laplace-approximated log-likelihood plus host-log aux."""
        log_lik, inner_eval_aux = self._compute_log_likelihood_impl(
            ct_params,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            obs_mask=obs_mask,
            extra_params=extra_params,
            latent_mode_init=latent_mode_init,
            transition_inputs=transition_inputs,
            include_aux=True,
            allow_stateful_cache=True,
        )
        assert inner_eval_aux is not None
        return log_lik, inner_eval_aux


__all__ = [
    "LaplaceLikelihood",
    # point.py re-exports
    "_build_linear_summary_augmented_system",
    "_dense_support_laplace_log_lik",
    "_ieks_smooth",
    "_linear_summary_augmented_ieks_laplace",
    "_point_ieks_mode",
    "_point_laplace_from_mode",
    # shared.py re-exports
    "_block_banded_logdet",
    "_build_ieks_system_from_prior",
    "_build_linear_summary_accumulator_plan",
    "_build_prior_tridiagonal_system",
    "_compute_profile_lower_bandwidths",
    "_factor_block_banded_cholesky",
    "_factor_block_profile_cholesky",
    "_infer_support_groups",
    "_predictive_latent_init",
    "_should_use_dense_support_laplace",
    "_solve_block_banded_from_cholesky",
    "_solve_block_profile_from_cholesky",
    "_solve_block_tridiagonal",
    "_tree_contains_tracer",
    "block_profile_logdet_packed_cotangent",
    # support.py re-exports
    "_assemble_support_aware_observation_system",
    "_make_support_window_derivatives",
    "_support_aware_ieks_laplace",
    "_support_aware_ieks_mode",
    "_support_aware_laplace_from_mode",
    "_support_aware_step_halving_search",
]
