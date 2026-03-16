"""Stage 4 Assembly Validation.

Shared compile + prior-predictive validation pipeline used by both
``stage4_grounding()`` (interactive) and ``stage4_orchestrated_flow()`` (batch).

The two paths differ only in their failure policy — domain logic is defined
once here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from causal_ssm_agent.flows import get_prefect_logger

if TYPE_CHECKING:
    import polars as pl

logger = get_prefect_logger(__name__)


@dataclass
class AssemblyValidation:
    """Result of stage 4 assembly validation (compile + prior predictive)."""

    compile_ok: bool = True
    compile_error: str | None = None
    pp_checked: bool = False
    pp_valid: bool = True
    pp_results: list = field(default_factory=list)  # list[PriorValidationResult]
    pp_raw_samples: Any = None

    @property
    def is_valid(self) -> bool:
        return self.compile_ok and self.pp_valid


def validate_assembly(
    model_spec: dict,
    priors: dict | None,
    raw_data: pl.DataFrame | None,
    causal_spec: dict | None,
) -> AssemblyValidation:
    """Validate stage 4 assembly: compile check + prior predictive.

    This is the single source of truth for the validation sequence.
    Both ``stage4_grounding()`` and ``validate_priors_task()`` use this.

    Steps:
        1. Compile check: trial compile (no priors) or real compile (with priors)
        2. Prior predictive validation (only when priors + raw_data present)

    Returns:
        AssemblyValidation with structured results.
    """
    from causal_ssm_agent.models.ssm_compiler import (
        compile_ssm_artifact,
        trial_compile_model_spec,
    )

    # Step 1: Compile check
    if priors:
        try:
            compile_ssm_artifact(model_spec, priors, causal_spec=causal_spec)
        except Exception as e:
            return AssemblyValidation(compile_ok=False, compile_error=str(e))
    else:
        compile_error = trial_compile_model_spec(model_spec, causal_spec)
        if compile_error:
            return AssemblyValidation(compile_ok=False, compile_error=compile_error)

    # Step 2: Prior predictive validation (only with real priors + data)
    if priors and raw_data is not None:
        from causal_ssm_agent.models.prior_predictive import validate_prior_predictive

        is_valid, results, raw_samples = validate_prior_predictive(
            model_spec, priors, raw_data, causal_spec=causal_spec
        )
        return AssemblyValidation(
            pp_checked=True,
            pp_valid=is_valid,
            pp_results=results,
            pp_raw_samples=raw_samples,
        )

    return AssemblyValidation()


def format_validation_feedback(
    validation: AssemblyValidation,
    priors: dict,
    changed_params: list[str] | None = None,
    data_stats: dict | None = None,
) -> str:
    """Format assembly validation result as feedback string.

    Used by ``stage4_grounding()`` to produce a single feedback string
    for the LLM.  The orchestrated flow formats per-parameter feedback
    separately for targeted re-elicitation.
    """
    if not validation.compile_ok:
        return f"COMPILE ERROR:\n{validation.compile_error}"

    if not validation.pp_checked or validation.pp_valid:
        return "VALID"

    from causal_ssm_agent.models.prior_predictive import format_parameter_feedback

    params = changed_params or list(priors.keys())
    parts = []
    for param_name in params:
        fb = format_parameter_feedback(
            parameter_name=param_name,
            results=validation.pp_results,
            prior=priors.get(param_name),
            data_stats=data_stats,
        )
        if fb:
            parts.append(fb)
    return "\n\n".join(parts) if parts else "PRIOR PREDICTIVE CHECK FAILED"
