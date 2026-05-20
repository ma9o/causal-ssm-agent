"""Continuous-time vector-field dynamics for SSMs.

This package owns the dynamics vocabulary and runtime objects:
vector-field components, composite drift specs, interventions,
stability checks, steady states, simulation, runtime envelopes, and
composite-spec serialization.

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
    default_linear_drift_spec,
    linear_drift_spec,
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
from .priors import (
    diagonal_decay_prior,
    effect_compartment_rate_prior,
    hill_ec50_prior,
    hill_emax_prior,
    hill_n_prior,
    linear_edge_weight_prior,
    multiplicative_weight_prior,
)
from .runtime import (
    RuntimeSSM,
    infer_linearisation,
    runtime_from_composite,
    runtime_from_dense_linear,
    runtime_from_ssm_model,
)
from .serialization import (
    compile_composite_from_dict,
    composite_spec_from_dict,
    composite_spec_to_dict,
    materialize_prior,
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
    "LinearEdge",
    "LinearEdgeSpec",
    "MultiplicativeEdge",
    "MultiplicativeEdgeSpec",
    "Override",
    "RuntimeSSM",
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
    "compile_composite_from_dict",
    "composite_spec_from_dict",
    "composite_spec_to_dict",
    "compute_steady_state",
    "constant_value",
    "default_linear_drift_spec",
    "diagonal_decay_prior",
    "effect_compartment_rate_prior",
    "hill_ec50_prior",
    "hill_emax_prior",
    "hill_n_prior",
    "infer_linearisation",
    "linear_drift_spec",
    "linear_edge_weight_prior",
    "linear_ramp",
    "materialize_prior",
    "multiplicative_weight_prior",
    "runtime_from_composite",
    "runtime_from_dense_linear",
    "runtime_from_ssm_model",
    "simulate",
    "simulate_pair",
]
