"""Continuous-time vector-field dynamics for SSMs.

This package owns the dynamics vocabulary:
vector-field components, composite drift specs, interventions,
stability checks, steady states, simulation, and composite-spec
serialization.

Block-level SSM parameter structure lives in ``ssm.structure``.
CT-to-DT matrix discretization lives in ``ssm.discretization``.
Prior/predictive validation lives in ``ssm.predictive``.
"""

from __future__ import annotations

from .composite import (
    CompiledComposite,
    ComponentSpec,
    CompositeSpec,
    DenseLinearSpec,
    DiagonalDecaySpec,
    HillEdgeSpec,
    InterceptSpec,
    LinearEdgeSpec,
    MultiplicativeEdgeSpec,
    StructuralDenseLinearSpec,
    StructuralInterceptSpec,
    compile_composite,
    iter_component_semantic_bindings,
)
from .edges import (
    DenseLinear,
    DiagonalDecay,
    DriftComponent,
    HillEdge,
    Intercept,
    LinearEdge,
    MultiplicativeEdge,
)
from .intervention import (
    EdgeInputOverride,
    Intervention,
    Override,
    ValueFn,
    VariableOverride,
    constant_value,
    linear_ramp,
)
from .linearisation import Linearisation, infer_linearisation
from .posterior import (
    PosteriorDynamicsSamples,
    component_param_samples_from_site_samples,
    posterior_dynamics_from_result,
    posterior_dynamics_from_samples,
)
from .priors import (
    diagonal_decay_prior,
    effect_compartment_rate_prior,
    hill_ec50_prior,
    hill_emax_prior,
    hill_n_prior,
    linear_edge_weight_prior,
    multiplicative_weight_prior,
)
from .serialization import (
    composite_spec_from_dict,
    composite_spec_to_dict,
)
from .simulator import SimulationConfig, simulate, simulate_pair
from .stability import StabilityReport, check_jacobian_stability
from .steady_state import compute_steady_state
from .vector_field import CompositeVectorField, VectorField, VectorFieldArgs

__all__ = [
    "CompiledComposite",
    "ComponentSpec",
    "CompositeSpec",
    "CompositeVectorField",
    "DenseLinear",
    "DenseLinearSpec",
    "DiagonalDecay",
    "DiagonalDecaySpec",
    "DriftComponent",
    "EdgeInputOverride",
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
    "Override",
    "PosteriorDynamicsSamples",
    "SimulationConfig",
    "StabilityReport",
    "StructuralDenseLinearSpec",
    "StructuralInterceptSpec",
    "ValueFn",
    "VariableOverride",
    "VectorField",
    "VectorFieldArgs",
    "check_jacobian_stability",
    "compile_composite",
    "composite_spec_from_dict",
    "composite_spec_to_dict",
    "component_param_samples_from_site_samples",
    "compute_steady_state",
    "constant_value",
    "diagonal_decay_prior",
    "effect_compartment_rate_prior",
    "hill_ec50_prior",
    "hill_emax_prior",
    "hill_n_prior",
    "infer_linearisation",
    "iter_component_semantic_bindings",
    "linear_edge_weight_prior",
    "linear_ramp",
    "multiplicative_weight_prior",
    "posterior_dynamics_from_result",
    "posterior_dynamics_from_samples",
    "simulate",
    "simulate_pair",
]
