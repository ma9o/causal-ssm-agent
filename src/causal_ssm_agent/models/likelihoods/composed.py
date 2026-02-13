"""Composed likelihood backend: Kalman sub-model + Particle sub-model.

When first-pass Rao-Blackwellization identifies a decoupled linear-Gaussian
sub-block, ComposedLikelihood runs the Kalman filter on that sub-block and
the particle filter on the remainder. The total log-likelihood is the sum
of the two sub-log-likelihoods (exact because the blocks are independent).

Parameter extraction uses the partition's numpy index arrays with jnp.ix_()
for matrix sub-block selection. These are compile-time constants (static
shapes), so JIT compiles efficiently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from causal_ssm_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
)

if TYPE_CHECKING:
    from causal_ssm_agent.models.likelihoods.graph_analysis import RBPartition


class ComposedLikelihood:
    """Two-level likelihood: exact Kalman on decoupled Gaussian block + PF on the rest.

    Args:
        partition: RBPartition from first-pass analysis.
        kalman_backend: KalmanLikelihood for the Gaussian sub-block.
        particle_backend: ParticleLikelihood for the non-Gaussian sub-block.
    """

    def __init__(self, partition: RBPartition, kalman_backend, particle_backend):
        self.partition = partition
        self.kalman_backend = kalman_backend
        self.particle_backend = particle_backend

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
        extra_params: dict | None = None,
    ) -> float:
        """Compute log-likelihood as sum of Kalman and particle sub-log-likelihoods.

        Extracts sub-model parameters using the partition indices, runs each
        backend independently, and returns their sum.
        """
        p = self.partition
        ki = jnp.array(p.kalman_idx)
        pi = jnp.array(p.particle_idx)
        oki = jnp.array(p.obs_kalman_idx)
        opi = jnp.array(p.obs_particle_idx)

        # --- Extract Kalman sub-model ---
        ct_kalman = CTParams(
            drift=ct_params.drift[jnp.ix_(ki, ki)],
            diffusion_cov=ct_params.diffusion_cov[jnp.ix_(ki, ki)],
            cint=ct_params.cint[ki] if ct_params.cint is not None else None,
        )
        meas_kalman = MeasurementParams(
            lambda_mat=measurement_params.lambda_mat[jnp.ix_(oki, ki)],
            manifest_means=measurement_params.manifest_means[oki],
            manifest_cov=measurement_params.manifest_cov[jnp.ix_(oki, oki)],
        )
        init_kalman = InitialStateParams(
            mean=initial_state.mean[ki],
            cov=initial_state.cov[jnp.ix_(ki, ki)],
        )
        obs_kalman = observations[:, oki]

        # --- Extract Particle sub-model ---
        ct_particle = CTParams(
            drift=ct_params.drift[jnp.ix_(pi, pi)],
            diffusion_cov=ct_params.diffusion_cov[jnp.ix_(pi, pi)],
            cint=ct_params.cint[pi] if ct_params.cint is not None else None,
        )
        meas_particle = MeasurementParams(
            lambda_mat=measurement_params.lambda_mat[jnp.ix_(opi, pi)],
            manifest_means=measurement_params.manifest_means[opi],
            manifest_cov=measurement_params.manifest_cov[jnp.ix_(opi, opi)],
        )
        init_particle = InitialStateParams(
            mean=initial_state.mean[pi],
            cov=initial_state.cov[jnp.ix_(pi, pi)],
        )
        obs_particle = observations[:, opi]

        # --- Handle obs_mask splitting ---
        obs_mask_kalman = None
        obs_mask_particle = None
        if obs_mask is not None:
            obs_mask_kalman = obs_mask[:, oki]
            obs_mask_particle = obs_mask[:, opi]

        # --- Compute sub-log-likelihoods ---
        ll_kalman = self.kalman_backend.compute_log_likelihood(
            ct_kalman,
            meas_kalman,
            init_kalman,
            obs_kalman,
            time_intervals,
            obs_mask=obs_mask_kalman,
        )

        ll_particle = self.particle_backend.compute_log_likelihood(
            ct_particle,
            meas_particle,
            init_particle,
            obs_particle,
            time_intervals,
            obs_mask=obs_mask_particle,
            extra_params=extra_params,
        )

        return ll_kalman + ll_particle
