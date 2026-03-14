"""Utility functions for causal-ssm-agent.

Heavy submodules (parametric_id, parametric_id_postfit) are imported lazily
to avoid circular imports with models/orchestrator.
"""

import importlib as _importlib

_LAZY = {
    "OutputSensitivityResult": "causal_ssm_agent.utils.parametric_id",
    "ProfileLikelihoodResult": "causal_ssm_agent.utils.parametric_id",
    "SBCResult": "causal_ssm_agent.utils.parametric_id",
    "TRuleResult": "causal_ssm_agent.utils.parametric_id",
    "check_t_rule": "causal_ssm_agent.utils.parametric_id",
    "count_free_params": "causal_ssm_agent.utils.parametric_id",
    "output_sensitivity_analysis": "causal_ssm_agent.utils.parametric_id",
    "profile_likelihood": "causal_ssm_agent.utils.parametric_id",
    "sbc_check": "causal_ssm_agent.utils.parametric_id",
    "simulate_ssm": "causal_ssm_agent.utils.parametric_id",
    "PowerScalingResult": "causal_ssm_agent.utils.parametric_id_postfit",
    "power_scaling_sensitivity": "causal_ssm_agent.utils.parametric_id_postfit",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        mod = _importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
