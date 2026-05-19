"""Pydantic models for inference diagnostics (MCMC, SVI, LOO, posterior).

These are the typed schemas for Stage 5 diagnostic payloads. They mirror
the dict structures already produced by InferenceResult.get_*_diagnostics()
and InferenceResult.get_posterior_*() methods, making them the source of
truth for the generated TypeScript types.
"""

from __future__ import annotations

from pydantic import BaseModel

from nof1_causal_lab.measurement_types import (
    AggregationFunction as _AggregationFunction,
)
from nof1_causal_lab.measurement_types import (
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
