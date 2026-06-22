"""Transition matrix construction for inference backends."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.discretization import (
    discretize_at_states_batched,
    discretize_linear_system_exact,
    discretize_system_with_inputs_batched,
)
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.linearisation import infer_linearisation
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorFieldArgs
from nof1_causal_lab.models.ssm.inference.targets.affine import derive_affine_dynamics
from nof1_causal_lab.models.ssm.inference.targets.base import RuntimeDynamics
from nof1_causal_lab.models.ssm.shapes import Array, Float


class DiscreteTransitionParams(NamedTuple):
    """Batched discrete-time transition parameters."""

    Ad: Float[Array, "T D D"]
    Qd: Float[Array, "T D D"]
    cd: Float[Array, "T D"] | None


def _continuous_input_forcing(
    dynamics: RuntimeDynamics,
    transition_inputs: Array | None,
    time_intervals: Float[Array, " T"],
) -> Float[Array, "T D"] | None:
    input_effect = dynamics.input_effect
    if input_effect is None or input_effect.shape[1] == 0:
        return None
    if transition_inputs is None:
        raise ValueError("SSM has known input effects but transition_inputs was not provided.")

    transition_inputs = jnp.asarray(transition_inputs, dtype=dynamics.diffusion_cov.dtype)
    expected_shape = (time_intervals.shape[0], input_effect.shape[1])
    if transition_inputs.shape != expected_shape:
        raise ValueError(
            f"transition_inputs must have shape {expected_shape}, got {transition_inputs.shape}"
        )

    return transition_inputs @ jnp.asarray(input_effect, dtype=dynamics.diffusion_cov.dtype).T


def build_discrete_transitions(
    dynamics: RuntimeDynamics,
    time_intervals: Float[Array, " T"],
    *,
    linearization_states: Array | None = None,
    transition_inputs: Array | None = None,
    intervention: Intervention | None = None,
) -> DiscreteTransitionParams:
    """Build batched CT-to-DT transitions from runtime vector-field dynamics.

    Constant-Jacobian vector fields use the exact affine fast path. Trajectory-
    dependent vector fields require one linearization state per interval and
    discretize the local affine system at each state.
    """
    time_intervals = jnp.asarray(time_intervals)
    if infer_linearisation(dynamics.vector_field) == "constant":
        affine_dynamics = derive_affine_dynamics(dynamics)
        Ad, Qd, cd = discretize_system_with_inputs_batched(
            affine_dynamics.drift,
            affine_dynamics.diffusion_cov,
            affine_dynamics.cint,
            affine_dynamics.input_effect,
            transition_inputs,
            time_intervals,
        )
        return DiscreteTransitionParams(Ad=Ad, Qd=Qd, cd=cd)

    if linearization_states is None:
        raise ValueError(
            "Trajectory-dependent vector-field discretization requires "
            "linearization_states with one state per interval."
        )

    linearization_states = jnp.asarray(linearization_states, dtype=dynamics.diffusion_cov.dtype)
    expected_shape = (time_intervals.shape[0], dynamics.vector_field.n_latent)
    if linearization_states.shape != expected_shape:
        raise ValueError(
            "linearization_states must have shape "
            f"{expected_shape}, got {linearization_states.shape}"
        )

    if intervention is None:
        intervention = Intervention.none()
    args = VectorFieldArgs(params=dynamics.vf_params, intervention=intervention)
    continuous_forcing = _continuous_input_forcing(
        dynamics,
        transition_inputs,
        time_intervals,
    )

    if continuous_forcing is None:
        Ad, Qd, cd = discretize_at_states_batched(
            dynamics.vector_field,
            linearization_states,
            args,
            dynamics.diffusion_cov,
            time_intervals,
        )
        return DiscreteTransitionParams(Ad=Ad, Qd=Qd, cd=cd)

    def _per_step(
        x_lin: Float[Array, " D"],
        dt: Array,
        forcing: Float[Array, " D"],
    ) -> tuple[Float[Array, "D D"], Float[Array, "D D"], Array | None]:
        drift, cint = dynamics.vector_field.linearize(x_lin, args)
        return discretize_linear_system_exact(
            drift,
            dynamics.diffusion_cov,
            cint + forcing,
            dt,
        )

    Ad, Qd, cd = jax.vmap(_per_step)(linearization_states, time_intervals, continuous_forcing)
    return DiscreteTransitionParams(Ad=Ad, Qd=Qd, cd=cd)
