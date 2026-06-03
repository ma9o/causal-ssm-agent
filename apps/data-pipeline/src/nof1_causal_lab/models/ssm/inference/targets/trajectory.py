"""Concrete latent trajectory targets for SSM inference runtimes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla

import nof1_causal_lab.models.ssm.inference.targets.euler_maruyama as euler_maruyama
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
    _predictive_latent_init,
    build_gaussian_trajectory_prior_terms,
    trajectory_prior_log_prob_from_terms,
)
from nof1_causal_lab.models.ssm.transition_kinds import (
    LATENT_TRANSITION_EULER_MARUYAMA,
    LATENT_TRANSITION_LOCAL_LINEAR_GAUSSIAN,
)

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class LocalLinearizationTarget:
    """Local-linearization (Van Loan) discretization of the nonlinear vector field.

    Linearizes the drift at each interval's state and applies the exact affine
    discretization, yielding affine-Gaussian transitions with closed-form prefix
    marginals. This is a discretization *scheme* an inference method requests, not
    a property of the model.
    """

    kind: str = LATENT_TRANSITION_LOCAL_LINEAR_GAUSSIAN
    supports_affine_prefix_marginals: bool = True
    jitter: float = 1e-6

    def initial_moments(self, context) -> tuple[jnp.ndarray, jnp.ndarray]:
        init_pred_mean = context.Ad[0] @ context.init_mean + context.cd[0]
        init_pred_cov = context.Ad[0] @ context.init_cov @ context.Ad[0].T + context.Qd[0]
        return init_pred_mean, symmetrize_with_jitter(init_pred_cov, jitter=self.jitter)

    def initial_log_prob(self, context, particle0: jnp.ndarray) -> jnp.ndarray:
        mean, cov = self.initial_moments(context)
        return _gaussian_log_prob_chol(
            particle0,
            mean,
            jnp.linalg.cholesky(symmetrize_with_jitter(cov, jitter=self.jitter)),
        )

    def predictive_latent_init(self, context) -> jnp.ndarray:
        return _predictive_latent_init(context.Ad, context.cd, context.init_mean)

    def sample_initial(
        self,
        key: jnp.ndarray,
        context,
        *,
        sample_shape: tuple[int, ...],
    ) -> jnp.ndarray:
        mean, cov = self.initial_moments(context)
        chol = jnp.linalg.cholesky(symmetrize_with_jitter(cov, jitter=self.jitter))
        eps = random.normal(key, (*sample_shape, mean.shape[-1]), dtype=mean.dtype)
        return mean + eps @ chol.T

    def sample_transition(
        self,
        key: jnp.ndarray,
        context,
        previous_states: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        mean = previous_states @ context.Ad[time_idx].T + context.cd[time_idx]
        chol = jnp.linalg.cholesky(symmetrize_with_jitter(context.Qd[time_idx], jitter=self.jitter))
        eps = random.normal(key, mean.shape, dtype=mean.dtype)
        return mean + eps @ chol.T

    def transition_log_prob(
        self,
        context,
        previous_state: jnp.ndarray,
        current_state: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        mean = previous_state @ context.Ad[time_idx].T + context.cd[time_idx]
        chol = jnp.linalg.cholesky(symmetrize_with_jitter(context.Qd[time_idx], jitter=self.jitter))
        return _gaussian_log_prob_chol(current_state, mean, chol)

    def transition_log_probs_for_pairs(
        self,
        context,
        previous_states: jnp.ndarray,
        current_states: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(
            lambda previous_state, current_state: self.transition_log_prob(
                context,
                previous_state,
                current_state,
                time_idx,
            )
        )(previous_states, current_states)

    def pairwise_transition_log_probs(
        self,
        context,
        previous_states: jnp.ndarray,
        current_states: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        means = previous_states @ context.Ad[time_idx].T + context.cd[time_idx]
        chol = jnp.linalg.cholesky(symmetrize_with_jitter(context.Qd[time_idx], jitter=self.jitter))
        diff = current_states[None, :, :] - means[:, None, :]
        flat = diff.reshape((-1, diff.shape[-1]))
        whitened = jla.solve_triangular(chol, flat.T, lower=True).T
        quadratic = jnp.sum(whitened * whitened, axis=-1).reshape(diff.shape[:2])
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        dim = previous_states.shape[-1]
        return -0.5 * (dim * _LOG_2PI + logdet + quadratic)

    def trajectory_prior_log_prob(
        self,
        context,
        latent_trajectory: jnp.ndarray,
        prior_terms: Any | None = None,
    ) -> jnp.ndarray:
        if prior_terms is None:
            prior_terms = build_gaussian_trajectory_prior_terms(
                context.Ad,
                context.Qd,
                context.cd,
                context.init_mean,
                context.init_cov,
                jitter=self.jitter,
            )
        return trajectory_prior_log_prob_from_terms(
            latent_trajectory,
            context.Ad,
            context.cd,
            prior_terms,
        )


@dataclass(frozen=True)
class EulerMaruyamaTarget:
    """Euler-Maruyama discretization of the nonlinear vector field.

    First-order scheme with sample-able transitions and closed-form density and
    gradients. This is a discretization *scheme* an inference method requests, not
    a property of the model.
    """

    vector_field: Any
    kind: str = LATENT_TRANSITION_EULER_MARUYAMA
    supports_affine_prefix_marginals: bool = False

    def initial_moments(self, context) -> tuple[jnp.ndarray, jnp.ndarray]:
        return euler_maruyama.initial_moments(self.vector_field, context)

    def initial_log_prob(self, context, particle0: jnp.ndarray) -> jnp.ndarray:
        return euler_maruyama.initial_log_prob(self.vector_field, context, particle0)

    def predictive_latent_init(self, context) -> jnp.ndarray:
        return euler_maruyama.predictive_latent_init(self.vector_field, context)

    def sample_initial(
        self,
        key: jnp.ndarray,
        context,
        *,
        sample_shape: tuple[int, ...],
    ) -> jnp.ndarray:
        mean, cov = self.initial_moments(context)
        chol = jnp.linalg.cholesky(cov)
        eps = random.normal(key, (*sample_shape, mean.shape[-1]), dtype=mean.dtype)
        return mean + eps @ chol.T

    def sample_transition(
        self,
        key: jnp.ndarray,
        context,
        previous_states: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        mean = jax.vmap(
            lambda previous_state: euler_maruyama.transition_mean(
                self.vector_field,
                context,
                previous_state,
                time_idx,
            )
        )(previous_states)
        chol = jnp.linalg.cholesky(euler_maruyama.transition_cov(context, time_idx))
        eps = random.normal(key, mean.shape, dtype=mean.dtype)
        return mean + eps @ chol.T

    def transition_log_prob(
        self,
        context,
        previous_state: jnp.ndarray,
        current_state: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        return euler_maruyama.transition_log_prob(
            self.vector_field,
            context,
            previous_state,
            current_state,
            time_idx,
        )

    def transition_log_probs_for_pairs(
        self,
        context,
        previous_states: jnp.ndarray,
        current_states: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        return euler_maruyama.transition_log_probs_for_pairs(
            self.vector_field,
            context,
            previous_states,
            current_states,
            time_idx,
        )

    def pairwise_transition_log_probs(
        self,
        context,
        previous_states: jnp.ndarray,
        current_states: jnp.ndarray,
        time_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        return euler_maruyama.pairwise_transition_log_probs(
            self.vector_field,
            context,
            previous_states,
            current_states,
            time_idx,
        )

    def trajectory_prior_log_prob(
        self,
        context,
        latent_trajectory: jnp.ndarray,
        prior_terms: Any | None = None,
    ) -> jnp.ndarray:
        del prior_terms
        return euler_maruyama.trajectory_prior_log_prob(
            self.vector_field,
            context,
            latent_trajectory,
        )


def _gaussian_log_prob_chol(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    chol: jnp.ndarray,
) -> jnp.ndarray:
    residual = value - mean
    whitened = jla.solve_triangular(chol, residual, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    dim = value.shape[-1]
    return -0.5 * (dim * _LOG_2PI + logdet + jnp.sum(whitened * whitened))
