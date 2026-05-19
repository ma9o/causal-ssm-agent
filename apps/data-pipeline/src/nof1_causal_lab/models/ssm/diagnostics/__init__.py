"""Parametric diagnostics for state-space models."""

from nof1_causal_lab.models.ssm.diagnostics.context import (
    ParametricIdContext,
    clear_diagnostics_sweep_context_cache,
    get_diagnostics_sweep_context,
)
from nof1_causal_lab.models.ssm.diagnostics.map_geometry import map_geometry_analysis
from nof1_causal_lab.models.ssm.diagnostics.power_scaling import power_scaling_sensitivity
from nof1_causal_lab.models.ssm.diagnostics.profile_likelihood import profile_likelihood
from nof1_causal_lab.models.ssm.diagnostics.results import (
    MAPCurvatureResult,
    MAPGeometryResult,
    MAPOptimizationRun,
    OutputSensitivityResult,
    OutputSensitivityUnsupportedError,
    PowerScalingResult,
    ProfileLikelihoodResult,
    SBCResult,
)
from nof1_causal_lab.models.ssm.diagnostics.sbc import sbc_check
from nof1_causal_lab.models.ssm.diagnostics.sensitivity import output_sensitivity_analysis
from nof1_causal_lab.models.ssm.diagnostics.simulation import simulate_ssm

__all__ = [
    "OutputSensitivityResult",
    "OutputSensitivityUnsupportedError",
    "MAPCurvatureResult",
    "MAPGeometryResult",
    "MAPOptimizationRun",
    "ParametricIdContext",
    "PowerScalingResult",
    "ProfileLikelihoodResult",
    "SBCResult",
    "clear_diagnostics_sweep_context_cache",
    "get_diagnostics_sweep_context",
    "map_geometry_analysis",
    "output_sensitivity_analysis",
    "power_scaling_sensitivity",
    "profile_likelihood",
    "sbc_check",
    "simulate_ssm",
]
