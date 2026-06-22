"""Concrete latent trajectory targets for SSM inference runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random

import nof1_causal_lab.models.ssm.inference.targets.euler_maruyama as euler_maruyama
from nof1_causal_lab.models.ssm.transition_kinds import (
    LATENT_TRANSITION_EULER_MARUYAMA,
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
