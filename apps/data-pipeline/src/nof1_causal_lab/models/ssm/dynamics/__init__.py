"""Non-linear vector-field dynamics framework for the SSM.

This module is the **substrate**: it owns the type system for the SSM's
continuous-time dynamics. Both inference (``models/ssm/inference``) and
counterfactual queries (``models/ssm/counterfactual``) consume from here,
but neither depends on the other.

The original Phase 1 work lived under ``counterfactual/``. As the
framework grew (primitives, compilation, discretisation, priors,
stability), it accumulated under the wrong directory — inference
ended up importing from ``counterfactual.x`` for things that aren't
counterfactual-specific. This module corrects that: anything that
describes *how the system evolves* (vector fields, edges, interventions,
compilation, discretisation, priors, stability, integrators) lives here.

Public API:

- ``VectorField``, ``CompositeVectorField``, ``VectorFieldArgs`` — dynamics
- ``DriftComponent``, ``DenseLinear``, ``DiagonalDecay``, ``Intercept``,
  ``LinearEdge``, ``HillEdge``, ``MultiplicativeEdge`` — primitive vocabulary
- ``Intervention``, ``VariableOverride``, ``EdgeInputOverride``,
  ``constant_value``, ``linear_ramp`` — intervention DSL
- ``CompositeSpec``, ``ComponentSpec``, ``CompiledComposite`` +
  per-component specs, ``compile_composite`` — declarative spec compiler
- ``discretize_at_state``, ``discretize_at_states_batched``,
  ``make_filter_dynamics_callback`` — CT→DT bridge (cuthbert + dense path)
- Default priors for Hill / Multiplicative / EffectCompartment / DiagonalDecay
- ``StabilityReport``, ``check_jacobian_stability`` — Jacobian-eigenvalue check
- ``simulate``, ``simulate_pair``, ``SimulationConfig`` — Diffrax forward
- ``compute_steady_state`` — Optimistix root-finder
"""

from __future__ import annotations

from .blocks import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
    default_diffusion_block,
    default_input_effect_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
)
from .compilation import (
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
from .config import (
    compile_composite_from_dict,
    composite_spec_from_dict,
    composite_spec_to_dict,
    materialize_prior,
)
from .discretization import (
    discretize_at_state,
    discretize_at_states_batched,
    make_filter_dynamics_callback,
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
from .prior_predictive import (
    CompositeAssemblyValidation,
    CompositePriorPredictive,
    composite_per_t_log_likelihood,
    composite_posterior_predictive_check,
    sample_composite_posterior_predictive_observations,
    sample_composite_prior_predictive,
    sample_composite_prior_predictive_full,
    sample_observations_from_latents,
    validate_composite_assembly,
    validate_composite_dynamics,
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
from .simulator import SimulationConfig, simulate, simulate_pair
from .stability import StabilityReport, check_jacobian_stability
from .steady_state import compute_steady_state
from .vector_field import CompositeVectorField, VectorField, VectorFieldArgs

__all__ = [
    "RuntimeSSM",
    "CompiledComposite",
    "ComponentSpec",
    "CompositeAssemblyValidation",
    "CompositePriorPredictive",
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
    "DiffusionBlockSpec",
    "ManifestCholBlockSpec",
    "SparseMatrixBlockSpec",
    "SparseVectorBlockSpec",
    "T0CholBlockSpec",
    "StructuralDenseLinearSpec",
    "StructuralInterceptSpec",
    "Override",
    "SimulationConfig",
    "StabilityReport",
    "ValueFn",
    "VariableOverride",
    "VectorField",
    "VectorFieldArgs",
    "runtime_from_composite",
    "runtime_from_dense_linear",
    "runtime_from_ssm_model",
    "check_jacobian_stability",
    "compile_composite",
    "composite_per_t_log_likelihood",
    "composite_posterior_predictive_check",
    "compile_composite_from_dict",
    "composite_spec_from_dict",
    "composite_spec_to_dict",
    "compute_steady_state",
    "constant_value",
    "default_diffusion_block",
    "default_input_effect_block",
    "default_lambda_block",
    "default_linear_drift_spec",
    "default_manifest_chol_block",
    "default_manifest_means_block",
    "default_static_state_sd_block",
    "default_t0_chol_block",
    "default_t0_means_block",
    "diagonal_decay_prior",
    "discretize_at_state",
    "discretize_at_states_batched",
    "effect_compartment_rate_prior",
    "hill_ec50_prior",
    "hill_emax_prior",
    "hill_n_prior",
    "infer_linearisation",
    "linear_drift_spec",
    "linear_edge_weight_prior",
    "linear_ramp",
    "make_filter_dynamics_callback",
    "materialize_prior",
    "multiplicative_weight_prior",
    "sample_composite_posterior_predictive_observations",
    "sample_composite_prior_predictive",
    "sample_composite_prior_predictive_full",
    "sample_observations_from_latents",
    "simulate",
    "simulate_pair",
    "validate_composite_assembly",
    "validate_composite_dynamics",
]
