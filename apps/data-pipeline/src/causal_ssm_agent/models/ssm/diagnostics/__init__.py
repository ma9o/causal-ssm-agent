"""Parametric diagnostics for state-space models."""

from causal_ssm_agent.models.ssm.diagnostics.context import (
    ParametricIdContext,
    clear_stage4b_sweep_context_cache,
    get_stage4b_sweep_context,
)
from causal_ssm_agent.models.ssm.diagnostics.power_scaling import power_scaling_sensitivity
from causal_ssm_agent.models.ssm.diagnostics.profile_likelihood import profile_likelihood
from causal_ssm_agent.models.ssm.diagnostics.results import (
    OutputSensitivityResult,
    OutputSensitivityUnsupportedError,
    PowerScalingResult,
    ProfileLikelihoodResult,
    SBCResult,
    TRuleResult,
)
from causal_ssm_agent.models.ssm.diagnostics.sbc import sbc_check
from causal_ssm_agent.models.ssm.diagnostics.sensitivity import output_sensitivity_analysis
from causal_ssm_agent.models.ssm.diagnostics.simulation import simulate_ssm
from causal_ssm_agent.models.ssm.diagnostics.t_rule import check_t_rule, count_free_params

__all__ = [
    "OutputSensitivityResult",
    "OutputSensitivityUnsupportedError",
    "ParametricIdContext",
    "PowerScalingResult",
    "ProfileLikelihoodResult",
    "SBCResult",
    "TRuleResult",
    "check_t_rule",
    "clear_stage4b_sweep_context_cache",
    "count_free_params",
    "get_stage4b_sweep_context",
    "output_sensitivity_analysis",
    "power_scaling_sensitivity",
    "profile_likelihood",
    "sbc_check",
    "simulate_ssm",
]
