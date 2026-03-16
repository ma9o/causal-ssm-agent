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
    from collections.abc import Callable

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


def run_stage4_assembly(
    model_spec: dict,
    priors: dict | None,
    raw_data: pl.DataFrame | None,
    causal_spec: dict | None,
    on_failure: Callable[[AssemblyValidation], str],
) -> tuple[AssemblyValidation, str]:
    """Run the shared Stage 4 assembly pipeline with an injected failure policy."""

    validation = validate_assembly(model_spec, priors, raw_data, causal_spec)
    if validation.is_valid:
        return validation, "VALID"
    return validation, on_failure(validation)


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


def validate_spec(model_spec: dict, causal_spec: dict | None) -> str | None:
    """Run the compile-time Stage 4 spec validation owned by the compiler."""
    from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec

    return trial_compile_model_spec(model_spec, causal_spec)


def merge_priors(existing: dict[str, dict] | None, new: dict[str, dict] | None) -> dict[str, dict]:
    """Merge prior updates into the current Stage 4 state."""
    return {**(existing or {}), **(new or {})}


def validate_prior_proposals(priors: dict[str, dict] | None) -> dict[str, dict]:
    """Schema-validate prior proposals before Stage 4 assembly."""
    from causal_ssm_agent.workers.schemas_prior import PriorProposal

    validated: dict[str, dict] = {}
    for name, prior in (priors or {}).items():
        try:
            validated[name] = PriorProposal.model_validate(prior).model_dump(mode="json")
        except Exception as exc:
            raise ValueError(f"SCHEMA ERRORS for prior '{name}':\n- {exc}") from exc
    return validated


def build_prior_predictive_samples(
    validation: AssemblyValidation,
    model_spec: dict,
) -> dict[str, list[float]]:
    """Forward-simulate per-variable prior predictive samples for the web payload."""
    if not validation.pp_valid or not validation.pp_raw_samples:
        return {}

    try:
        import jax.numpy as jnp
        import numpy as np

        from causal_ssm_agent.models.posterior_predictive import (
            simulate_posterior_predictive,
        )
        from causal_ssm_agent.orchestrator.schemas_model import ModelSpec

        spec = ModelSpec.model_validate(model_spec) if isinstance(model_spec, dict) else model_spec
        manifest_names = [lik.variable for lik in spec.likelihoods]
        manifest_dists = [lik.distribution.value for lik in spec.likelihoods]
        manifest_links = [lik.link.value for lik in spec.likelihoods]

        y_sim = simulate_posterior_predictive(
            validation.pp_raw_samples,
            times=jnp.arange(30, dtype=jnp.float32),
            manifest_dists=manifest_dists,
            manifest_links=manifest_links,
            n_subsample=100,
        )
        y_np = np.asarray(y_sim)

        samples: dict[str, list[float]] = {}
        for idx, name in enumerate(manifest_names):
            col = y_np[:, :, idx].flatten()
            col = col[np.isfinite(col)]
            samples[name] = col.tolist()
        return samples
    except Exception as exc:
        logger.warning("Prior predictive simulation failed: %s", exc)
        return {}


def build_validation_payload(
    validation: AssemblyValidation,
    model_spec: dict,
) -> dict[str, Any]:
    """Convert ``AssemblyValidation`` into the web-facing validation payload."""
    if not validation.compile_ok:
        return {
            "is_valid": False,
            "results": [],
            "issues": [f"Compile error: {validation.compile_error}"],
            "prior_predictive_samples": {},
        }

    results = [result.model_dump() for result in validation.pp_results]
    return {
        "is_valid": validation.pp_valid,
        "results": results,
        "issues": [
            result.issue for result in validation.pp_results if not result.is_valid and result.issue
        ],
        "prior_predictive_samples": build_prior_predictive_samples(validation, model_spec),
    }


def build_retry_feedback(
    validation: AssemblyValidation,
    priors: dict[str, dict],
    *,
    causal_spec: dict | None = None,
    data_stats: dict | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Identify failed parameters and build per-parameter retry feedback."""
    from causal_ssm_agent.models.prior_predictive import (
        format_parameter_feedback,
        get_failed_parameters,
    )

    failed_param_names = get_failed_parameters(
        validation.pp_results,
        list(priors.keys()),
        causal_spec=causal_spec,
    )
    if not failed_param_names and not validation.pp_valid:
        failed_param_names = list(priors.keys())

    feedbacks = {
        param_name: format_parameter_feedback(
            parameter_name=param_name,
            results=validation.pp_results,
            prior=priors.get(param_name),
            data_stats=data_stats,
        )
        for param_name in failed_param_names
    }
    return failed_param_names, feedbacks


def compile_model_artifact(
    model_spec: dict,
    priors: dict[str, dict],
    raw_data: pl.DataFrame,
    causal_spec: dict | None = None,
) -> dict[str, Any]:
    """Compile and verify the executable SSM artifact for Stage 4 output."""
    from causal_ssm_agent.models.ssm_builder import build_ssm_builder
    from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact

    try:
        compiled_ssm = compile_ssm_artifact(model_spec, priors, causal_spec=causal_spec)
        builder = build_ssm_builder(
            raw_data=raw_data,
            compiled_ssm=compiled_ssm,
        )
        return {
            "model_built": True,
            "model_type": builder._model_type,
            "version": builder.version,
            "compiled_ssm": compiled_ssm,
        }
    except NotImplementedError:
        return {
            "model_built": False,
            "error": "SSM implementation not available",
        }
    except Exception as exc:
        return {
            "model_built": False,
            "error": str(exc),
        }


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
    details = "\n\n".join(parts) if parts else "PRIOR PREDICTIVE CHECK FAILED"
    return f"PRIOR PREDICTIVE FEEDBACK:\n{details}"
