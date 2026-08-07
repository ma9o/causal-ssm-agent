"""Continuous-time vector-field dynamics for SSMs.

This package owns the dynamics vocabulary:
vector-field components, dynamics specs, interventions, stability checks,
steady states, simulation, and dynamics-spec serialization.

Block-level SSM parameter structure lives in ``ssm.structure``.
CT-to-DT matrix discretization lives in ``ssm.discretization``.
Prior/predictive validation lives in ``ssm.predictive``.
"""

from __future__ import annotations

from nof1_causal_lab.models.ssm.structure.parameters import Fixed, Free, ParameterSlot

from .edges import (
    DiagonalDecay,
    HillEdge,
    Intercept,
    LinearEdge,
    MultiplicativeEdge,
    NodePotential,
    StateDecay,
    StateIntercept,
    VectorFieldComponent,
)
from .intervention import (
    EdgeInputOverride,
    Intervention,
    Override,
    PrecomputedValueFn,
    ValueFn,
    VariableOverride,
    constant_value,
    linear_ramp,
    precomputed_value,
)
from .linearisation import Linearisation, infer_linearisation
from .posterior import (
    PosteriorDynamicsSamples,
    component_param_samples_from_site_samples,
    posterior_dynamics_from_result,
    posterior_dynamics_from_samples,
)
from .runtime import (
    VectorFieldRuntime,
    build_vector_field_runtime_from_samples,
    pack_vector_field_params_from_samples,
    sample_vector_field_runtime,
)
from .serialization import (
    dynamics_spec_from_dict,
    dynamics_spec_to_dict,
)
from .simulator import SimulationConfig, simulate, simulate_pair
from .spec import (
    CompiledDynamics,
    ComponentSpec,
    DiagonalDecaySpec,
    DynamicsSpec,
    HillEdgeSpec,
    InterceptSpec,
    LinearEdgeSpec,
    MultiplicativeEdgeSpec,
    NodePotentialSpec,
    StateDecaySpec,
    StateInterceptSpec,
    compile_dynamics,
    iter_dynamics_semantic_bindings,
)
from .steady_state import compute_steady_state
from .vector_field import VectorField, VectorFieldArgs

__all__ = [
    "CompiledDynamics",
    "ComponentSpec",
    "DynamicsSpec",
    "DiagonalDecay",
    "DiagonalDecaySpec",
    "VectorFieldComponent",
    "EdgeInputOverride",
    "Fixed",
    "Free",
    "HillEdge",
    "HillEdgeSpec",
    "Intercept",
    "InterceptSpec",
    "Intervention",
    "Linearisation",
    "LinearEdge",
    "LinearEdgeSpec",
    "MultiplicativeEdge",
    "MultiplicativeEdgeSpec",
    "NodePotential",
    "NodePotentialSpec",
    "Override",
    "ParameterSlot",
    "PosteriorDynamicsSamples",
    "PrecomputedValueFn",
    "SimulationConfig",
    "StateDecay",
    "StateDecaySpec",
    "StateIntercept",
    "StateInterceptSpec",
    "ValueFn",
    "VariableOverride",
    "VectorField",
    "VectorFieldArgs",
    "VectorFieldRuntime",
    "build_vector_field_runtime_from_samples",
    "compile_dynamics",
    "dynamics_spec_from_dict",
    "dynamics_spec_to_dict",
    "component_param_samples_from_site_samples",
    "compute_steady_state",
    "constant_value",
    "infer_linearisation",
    "iter_dynamics_semantic_bindings",
    "linear_ramp",
    "pack_vector_field_params_from_samples",
    "posterior_dynamics_from_result",
    "posterior_dynamics_from_samples",
    "precomputed_value",
    "simulate",
    "simulate_pair",
    "sample_vector_field_runtime",
]
