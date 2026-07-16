"""Compile and materialize an admitted model specification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from nof1_causal_lab.models.compilation_errors import AggregatedCompileError

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.models.ssm.compile.contracts import CompiledSSMArtifact
    from nof1_causal_lab.workers.schemas_prior import PriorValidationResult

_RECOVERABLE_MODEL_SPEC_ASSEMBLY_ERRORS = (
    AggregatedCompileError,
    ValidationError,
    ValueError,
)


@dataclass
class AssemblyValidation:
    """Result of compile-only assembly validation."""

    normalized_statistical_model_spec: dict[str, Any] | None = None
    compile_ok: bool = True
    compile_error: str | None = None
    compiled_ssm: CompiledSSMArtifact | None = None
    diagnostics: list[PriorValidationResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.compile_ok

    @property
    def compile_diagnostics(self) -> list[PriorValidationResult]:
        return [d for d in self.diagnostics if d.origin == "compile"]


def validate_assembly(
    statistical_model_spec: dict,
    authored_priors: dict | None,
    causal_design: dict | None,
) -> AssemblyValidation:
    """Compile authored inputs and retain compiler-owned diagnostics.

    Construct admission and the full-model barrier own statistical validation.
    Assembly intentionally cannot invoke the removed legacy whole-model PPC suite.
    """
    from nof1_causal_lab.models.ssm.compile.artifact import (
        compile_ssm_artifact,
        trial_compile_statistical_model_spec,
    )

    candidate = _prepare_statistical_model_spec(statistical_model_spec)
    if authored_priors:
        try:
            compiled_ssm = compile_ssm_artifact(
                candidate, authored_priors, causal_design=causal_design
            )
        except _RECOVERABLE_MODEL_SPEC_ASSEMBLY_ERRORS as exc:
            return AssemblyValidation(
                normalized_statistical_model_spec=candidate,
                compile_ok=False,
                compile_error=str(exc),
                diagnostics=_collect_compile_failure_diagnostics(exc),
            )
        compile_diagnostics = _collect_compile_diagnostics(compiled_ssm)
    else:
        compile_error = trial_compile_statistical_model_spec(candidate, causal_design)
        if compile_error:
            return AssemblyValidation(
                normalized_statistical_model_spec=candidate,
                compile_ok=False,
                compile_error=str(compile_error),
                diagnostics=_collect_compile_failure_diagnostics(compile_error),
            )
        compiled_ssm = None
        compile_diagnostics = []

    return AssemblyValidation(
        normalized_statistical_model_spec=candidate,
        diagnostics=compile_diagnostics,
        compiled_ssm=compiled_ssm,
    )


def _collect_compile_failure_diagnostics(failure: Any) -> list[PriorValidationResult]:
    """Best-effort extraction of structured diagnostics from a compile failure payload."""
    from nof1_causal_lab.workers.schemas_prior import PriorValidationResult

    pending: list[Any] = [failure]
    seen_ids: set[int] = set()
    typed: list[PriorValidationResult] = []

    while pending:
        candidate = pending.pop(0)
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)

        if isinstance(candidate, PriorValidationResult):
            typed.append(candidate)
            continue

        if isinstance(candidate, dict):
            if "compile_diagnostics" in candidate:
                pending.append(candidate.get("compile_diagnostics"))
                continue
            try:
                typed.append(PriorValidationResult.model_validate(candidate))
                continue
            except ValidationError:
                pass

        model_dump = getattr(candidate, "model_dump", None)
        if callable(model_dump):
            pending.append(model_dump(mode="json"))
            continue

        if isinstance(candidate, (list, tuple, set, frozenset)):
            pending.extend(candidate)
            continue

        for attr_name in ("compile_diagnostics", "diagnostics", "errors", "results"):
            attr_value = getattr(candidate, attr_name, None)
            if attr_value is not None:
                pending.append(attr_value)

    return typed


def _prepare_statistical_model_spec(statistical_model_spec: dict) -> dict[str, Any]:
    """Normalize a model-spec statistical model spec before any compile-time work."""
    return deepcopy(statistical_model_spec)


def _collect_compile_diagnostics(
    compiled_ssm: CompiledSSMArtifact,
) -> list[PriorValidationResult]:
    """Collect typed compiler-owned diagnostics for model-spec feedback."""
    return compiled_ssm.compile_diagnostics


def build_exact_prior_predictive_samples(
    compiled_ssm: CompiledSSMArtifact,
    data_for_model: pl.DataFrame,
    *,
    n_draws: int = 200,
) -> dict[str, list[float]]:
    """Simulate the admitted full model once for the persisted Data-vs-Prior view."""
    import numpy as np

    from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime, sample_prior_predictive

    runtime = prepare_model_runtime(data_for_model, compiled_ssm=compiled_ssm)
    observation_mask = np.isfinite(np.asarray(runtime.observations))
    predictive = sample_prior_predictive(
        runtime.model,
        samples=n_draws,
        times=runtime.times,
        observation_support=runtime.observation_support,
        observation_mask=observation_mask,
        transition_inputs=runtime.transition_inputs,
    )
    observations = np.asarray(predictive["observations"])
    effective_mask = np.asarray(predictive["observations_mask"], dtype=bool)
    return {
        name: observations[:, :, index][effective_mask[:, :, index]].tolist()
        for index, name in enumerate(runtime.manifest_names)
    }


def _collect_validation_warning_messages(validation: AssemblyValidation) -> list[str]:
    """Flatten warning diagnostics into user-facing text."""
    messages = [
        result.issue
        for result in validation.diagnostics
        if result.severity == "warning" and result.issue
    ]
    return [message for message in messages if isinstance(message, str)]


def compile_model_artifact(
    statistical_model_spec: dict,
    authored_priors: dict[str, dict],
    data_for_model: pl.DataFrame,
    causal_design: dict | None = None,
    compiled_ssm: CompiledSSMArtifact | None = None,
) -> dict[str, Any]:
    """Compile and verify the executable SSM artifact for model-spec output."""
    from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
    from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime

    try:
        artifact = compiled_ssm or compile_ssm_artifact(
            _prepare_statistical_model_spec(statistical_model_spec),
            authored_priors,
            causal_design=causal_design,
        )
    except _RECOVERABLE_MODEL_SPEC_ASSEMBLY_ERRORS as exc:
        return {
            "model_built": False,
            "error": str(exc),
        }

    try:
        prepare_model_runtime(data_for_model, compiled_ssm=artifact)
        return {
            "model_built": True,
            "model_type": "SSM",
            "compiled_ssm": artifact,
        }
    except NotImplementedError:
        return {
            "model_built": False,
            "error": "SSM implementation not available",
            "compiled_ssm": artifact,
        }
    except _RECOVERABLE_MODEL_SPEC_ASSEMBLY_ERRORS as exc:
        return {
            "model_built": False,
            "error": str(exc),
            "compiled_ssm": artifact,
        }


def materialize_model_spec_result(
    *,
    statistical_model_spec: dict[str, Any],
    authored_priors: dict[str, dict],
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict[str, Any]] | None,
    causal_design: dict | None,
    validation: AssemblyValidation | None = None,
    search_queries: dict[str, str] | None = None,
    skip_ppc: bool = True,
) -> dict[str, Any]:
    """Build the persisted result from construct-admitted authored inputs.

    ``skip_ppc`` remains as a compatibility guard for the Temporal caller. The
    removed whole-model PPC suite cannot be requested; construct admission and the
    exact full-model barrier own validation.
    """
    from nof1_causal_lab.models.ssm.compile.artifact import resolve_prior_proposals

    if not skip_ppc:
        raise ValueError("Legacy whole-model prior-predictive validation has been removed.")

    validation = validation or validate_assembly(
        statistical_model_spec,
        authored_priors,
        causal_design,
    )
    del indicator_audits
    normalized_statistical_model_spec = (
        validation.normalized_statistical_model_spec or statistical_model_spec
    )
    model_result = compile_model_artifact(
        normalized_statistical_model_spec,
        authored_priors,
        data_for_model,
        causal_design=causal_design,
        compiled_ssm=validation.compiled_ssm,
    )
    compiled_ssm = model_result.pop("compiled_ssm", None)
    resolved_priors = (
        resolve_prior_proposals(compiled_ssm, authored_priors=authored_priors)
        if compiled_ssm
        else []
    )

    prior_predictive_samples = (
        build_exact_prior_predictive_samples(compiled_ssm, data_for_model)
        if compiled_ssm is not None
        else {}
    )

    result = {
        "statistical_model_spec": normalized_statistical_model_spec,
        "authored_priors": authored_priors,
        "resolved_priors": resolved_priors,
        "search_queries": search_queries or None,
        "validation_warnings": _collect_validation_warning_messages(validation) or None,
        "_causal_design": causal_design,
        "prior_predictive_samples": prior_predictive_samples,
        "prior_predictive_diagnostics": [],
    }
    if compiled_ssm is not None:
        result["_compiled_ssm"] = compiled_ssm
    return result
