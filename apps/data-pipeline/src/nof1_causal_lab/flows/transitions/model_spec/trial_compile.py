"""Application-owned trial compilation with explicit planned defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.prior_planning import build_default_prior_plan
from nof1_causal_lab.models.ssm.compile import artifact as ssm_compiler

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan


def trial_compile_statistical_model_spec(
    statistical_model_spec: StatisticalModelSpec,
    structural_plan: StructuralPlan,
) -> str | None:
    """Compile with an explicit default PriorPlan and return any structural error."""
    try:
        ssm_compiler.compile_ssm_artifact(
            statistical_model_spec,
            build_default_prior_plan(statistical_model_spec),
            structural_plan=structural_plan,
        )
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        return str(exc)
    return None


__all__ = ["trial_compile_statistical_model_spec"]
