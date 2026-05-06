"""Pydantic models for inference diagnostics (MCMC, SVI, LOO, posterior).

These are the typed schemas for Stage 5 diagnostic payloads. They mirror
the dict structures already produced by InferenceResult.get_*_diagnostics()
and InferenceResult.get_posterior_*() methods, making them the source of
truth for the generated TypeScript types.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from causal_ssm_agent.measurement_types import (
    AggregationFunction as _AggregationFunction,
)
from causal_ssm_agent.measurement_types import (
    MeasurementDtype as _MeasurementDtype,
)

AggregationFunction = _AggregationFunction
MeasurementDtype = _MeasurementDtype

# ---------------------------------------------------------------------------
# MCMC diagnostics
# ---------------------------------------------------------------------------


class MCMCParamDiagnostic(BaseModel):
    """Per-parameter MCMC convergence diagnostics."""

    parameter: str
    r_hat: float | list[float]
    ess_bulk: float | list[float]
    ess_tail: float | list[float] | None = None
    mcse_mean: float | list[float] | None = None


class TraceChain(BaseModel):
    """Thinned trace values for a single chain."""

    chain: int
    values: list[float]


class TraceData(BaseModel):
    """Per-parameter trace data across chains."""

    parameter: str
    chains: list[TraceChain]


class RankHistogramChain(BaseModel):
    """Rank histogram bin counts for a single chain."""

    chain: int
    counts: list[int]


class RankHistogram(BaseModel):
    """Per-parameter rank histogram for chain mixing assessment."""

    parameter: str
    n_bins: int
    expected_per_bin: float
    chains: list[RankHistogramChain]


class EnergyHistogram(BaseModel):
    """Histogram of energy values (bin centers + density)."""

    bin_centers: list[float]
    density: list[float]


class EnergyDiagnostics(BaseModel):
    """Hamiltonian energy diagnostics."""

    energy_hist: EnergyHistogram
    energy_transition_hist: EnergyHistogram
    bfmi: list[float]


class MCMCDiagnostics(BaseModel):
    """Top-level MCMC diagnostics container."""

    per_parameter: list[MCMCParamDiagnostic]
    num_divergences: int = 0
    divergence_rate: float = 0.0
    tree_depth_mean: float = 0.0
    tree_depth_max: int = 0
    accept_prob_mean: float = 0.0
    num_chains: int | None = None
    num_samples: int | None = None
    trace_data: list[TraceData] | None = None
    rank_histograms: list[RankHistogram] | None = None
    energy: EnergyDiagnostics | None = None


# ---------------------------------------------------------------------------
# SVI diagnostics
# ---------------------------------------------------------------------------


class SVIDiagnostics(BaseModel):
    """SVI (variational inference) diagnostics."""

    elbo_losses: list[float]


# ---------------------------------------------------------------------------
# SMC diagnostics
# ---------------------------------------------------------------------------


class SMCDiagnostics(BaseModel):
    """Sequential Monte Carlo diagnostics."""

    beta_schedule: list[float]
    ess_history: list[float]
    accept_rates: list[float]
    n_levels: int
    n_particles: int


# ---------------------------------------------------------------------------
# LOO diagnostics
# ---------------------------------------------------------------------------


class LOODiagnostics(BaseModel):
    """Leave-one-out cross-validation diagnostics (ArviZ).

    Uses one-step-ahead predictive log-likelihoods from the filter's
    innovation decomposition. Each LOO "observation" is one complete
    timestep (all manifest variables at time t), not individual cells.
    """

    elpd_loo: float
    p_loo: float
    se: float
    n_data_points: int
    observation_unit: str = "timestep"
    pareto_k: list[float] | None = None
    n_bad_k: int | None = None
    loo_pit: list[float] | None = None


# ---------------------------------------------------------------------------
# Posterior visualization data
# ---------------------------------------------------------------------------


class PosteriorMarginal(BaseModel):
    """Marginal posterior density for a single scalar parameter."""

    parameter: str
    x_values: list[float]
    density: list[float]
    mean: float
    sd: float
    hdi_3: float
    hdi_97: float


class PosteriorPair(BaseModel):
    """Pairwise posterior scatter data for joint visualization."""

    param_x: str
    param_y: str
    x_values: list[float]
    y_values: list[float]
    divergent: list[bool] | None = None


# ---------------------------------------------------------------------------
# Parametric identifiability result models
# ---------------------------------------------------------------------------


ParameterClassification = Literal[
    "identified",
    "practically_unidentifiable",
    "structurally_unidentifiable",
]


class ParameterIdentification(BaseModel):
    """Per-parameter identifiability classification."""

    name: str
    classification: ParameterClassification
    contraction_ratio: float | None = None
    profile_x: list[float] | None = None
    profile_ll: list[float] | None = None


class ParametricIdSummary(BaseModel):
    """Summary of parametric identifiability issues."""

    structural_issues: list[str] = Field(default_factory=list)
    boundary_issues: list[str] = Field(default_factory=list)
    weak_params: list[str] = Field(default_factory=list)


class SensitivityEntry(BaseModel):
    """Per-parameter output sensitivity analysis entry."""

    parameter: str
    interpretable_parameter: str
    sensitivity_norm: float
    effective_sv: float
    sv_status: Literal["pass", "warn", "fail"]
    normalized_effective_sv: float
    normalized_sv_status: Literal["pass", "warn", "fail"]
    identifiable: bool


class SensitivityDirectionLoading(BaseModel):
    """One parameter's loading within a weak local sensitivity direction."""

    parameter: str
    interpretable_parameter: str
    loading: float
    abs_loading: float


class SensitivityDirection(BaseModel):
    """A direction in parameter space from the normalized sensitivity SVD."""

    index: int
    singular_value: float
    normalized_singular_value: float
    status: Literal["pass", "warn", "fail"]
    top_loadings: list[SensitivityDirectionLoading]


class SensitivityAnalysisResult(BaseModel):
    """Output sensitivity analysis result (pre-inference identifiability).

    Structural identifiability check via the Jacobian of the forward model's
    emitted-observation moment summary. Near-zero singular values indicate
    parameter combinations that observations cannot distinguish.
    """

    singular_values: list[float]
    normalized_singular_values: list[float]
    deficiency_count: int
    weak_directions: list[SensitivityDirection]
    per_parameter: list[SensitivityEntry]
    n_draws: int
    n_observations: int
    n_parameters: int


class CurvatureParameterEntry(BaseModel):
    """Per-parameter local-curvature summary at the selected MAP."""

    parameter: str
    interpretable_parameter: str
    diagonal_curvature: float
    effective_eigenvalue: float
    status: Literal["pass", "warn", "fail"]
    normalized_effective_eigenvalue: float
    normalized_status: Literal["pass", "warn", "fail"]


class CurvatureDirectionLoading(BaseModel):
    """One parameter's loading within a weak local-curvature eigen-direction."""

    parameter: str
    interpretable_parameter: str
    loading: float
    abs_loading: float


class CurvatureDirection(BaseModel):
    """A weak Hessian eigen-direction within the MAP neighborhood."""

    index: int
    eigenvalue: float
    normalized_eigenvalue: float
    status: Literal["pass", "warn", "fail"]
    top_loadings: list[CurvatureDirectionLoading]


class MAPCurvatureResult(BaseModel):
    """One Hessian family's local geometry at the selected MAP."""

    eigenvalues: list[float]
    normalized_eigenvalues: list[float]
    negative_direction_count: int
    deficiency_count: int
    positive_definite: bool
    condition_number: float | None = None
    normalized_condition_number: float | None = None
    weak_directions: list[CurvatureDirection]
    per_parameter: list[CurvatureParameterEntry]


class MAPOptimizationRun(BaseModel):
    """One start in the multi-start MAP search."""

    index: int
    start_kind: str
    start_log_posterior: float
    log_posterior: float
    log_likelihood: float
    log_prior: float
    objective: float
    success: bool
    status: int
    message: str
    n_iters: int
    n_function_evals: int
    grad_norm: float
    distance_to_best: float


class MAPGeometryResult(BaseModel):
    """Dataset-conditioned MAP search plus H_lik / H_post local geometry."""

    n_starts: int
    n_successful_starts: int
    best_start_index: int
    map_log_posterior: float
    map_log_likelihood: float
    map_log_prior: float
    final_grad_norm: float
    runner_up_objective_gap: float | None = None
    starts: list[MAPOptimizationRun]
    likelihood_curvature: MAPCurvatureResult
    posterior_curvature: MAPCurvatureResult
    prior_rescued_parameters: list[str]
    boundary_parameters: list[str]


class ParametricIdResult(BaseModel):
    """Full parametric identifiability result (Stage 4b payload)."""

    checked: bool = False
    sensitivity_analysis: SensitivityAnalysisResult | None = None
    map_geometry: MAPGeometryResult | None = None
    summary: ParametricIdSummary | None = None
    per_param_classification: list[ParameterIdentification] | None = None
    threshold: float | None = None
    error: str | None = None


class InferenceStructureVariable(BaseModel):
    """A single latent or observed channel assignment in the active split."""

    name: str
    method: Literal["kalman", "particle"]


class FirstPassRBResult(BaseModel):
    """Active first-pass Rao-Blackwellization plan for the prepared runtime."""

    status: Literal["active", "inactive"]
    latent_variables: list[InferenceStructureVariable]
    obs_variables: list[InferenceStructureVariable]


class InferenceStructureResult(BaseModel):
    """Canonical inference-structure plan shared by pipeline and web."""

    likelihood_path: Literal["kalman", "composed", "particle"]
    auto_method: Literal["aux_gibbs"]
    first_pass_rb: FirstPassRBResult


# ---------------------------------------------------------------------------
# Treatment effects
# ---------------------------------------------------------------------------


class TemporalEffect(BaseModel):
    """Temporal decomposition of a treatment effect."""

    effect_1d: float
    effect_7d: float
    effect_30d: float
    peak_effect: float
    time_to_peak_days: float


# ---------------------------------------------------------------------------
# Named type aliases (formalized from inline constants)
# ---------------------------------------------------------------------------
