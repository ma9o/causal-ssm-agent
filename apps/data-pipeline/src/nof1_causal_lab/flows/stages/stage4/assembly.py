"""Stage 4 Assembly Validation.

Shared compile + prior-predictive + sensitivity validation pipeline used by both
``stage4_grounding()`` (interactive) and ``stage4_agentic_flow()`` (batch).

The two paths differ only in their failure policy - domain logic is defined
once here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.compilation_errors import AggregatedCompileError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl

    from nof1_causal_lab.workers.schemas_prior import PriorValidationResult

logger = get_prefect_logger(__name__)
_RECOVERABLE_STAGE4_ASSEMBLY_ERRORS = (
    AggregatedCompileError,
    ValidationError,
    ValueError,
)


@dataclass
class AssemblyValidation:
    """Result of stage 4 assembly validation."""

    normalized_model_spec: dict[str, Any] | None = None
    compile_ok: bool = True
    compile_error: str | None = None
    compiled_ssm: dict[str, Any] | None = None
    pp_checked: bool = False
    pp_valid: bool = True
    diagnostics: list[PriorValidationResult] = field(default_factory=list)
    pp_raw_samples: Any = None
    sensitivity_consulted: bool = False
    sensitivity_supported: bool = False
    sensitivity_valid: bool = True
    sensitivity_payload: dict[str, Any] | None = None
    sensitivity_warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.compile_ok and self.pp_valid and self.sensitivity_valid

    @property
    def compile_diagnostics(self) -> list[PriorValidationResult]:
        return [d for d in self.diagnostics if d.origin == "compile"]

    @property
    def prior_predictive_diagnostics(self) -> list[PriorValidationResult]:
        return [d for d in self.diagnostics if d.origin == "prior_predictive"]

    @property
    def has_sensitivity_failure(self) -> bool:
        return (
            self.sensitivity_consulted and self.sensitivity_supported and not self.sensitivity_valid
        )


def validate_assembly(
    model_spec: dict,
    authored_priors: dict | None,
    data_for_model: pl.DataFrame | None,
    indicator_audits: dict[str, dict[str, Any]] | None,
    causal_spec: dict | None,
    *,
    skip_ppc: bool = False,
) -> AssemblyValidation:
    """Validate stage 4 assembly: compile check + prior predictive + sensitivity.

    This is the single source of truth for the validation sequence.
    Both ``stage4_grounding()`` and ``stage4_agentic_flow()`` use this.

    Steps:
        1. Compile check: trial compile (no priors) or real compile (with priors)
        2. Prior predictive validation (only when authored priors + data_for_model present
           and skip_ppc is False)
        3. Jacobian sensitivity validation (only after compile + PPC succeed)

    Returns:
        AssemblyValidation with structured results.
    """
    from nof1_causal_lab.models.ssm_compiler import compile_ssm_artifact, trial_compile_model_spec

    candidate = _prepare_model_spec(model_spec)
    if authored_priors:
        try:
            compiled_ssm = compile_ssm_artifact(candidate, authored_priors, causal_spec=causal_spec)
        except _RECOVERABLE_STAGE4_ASSEMBLY_ERRORS as exc:
            return AssemblyValidation(
                normalized_model_spec=candidate,
                compile_ok=False,
                compile_error=str(exc),
                diagnostics=_collect_compile_failure_diagnostics(exc),
            )
        compile_diagnostics = _collect_compile_diagnostics(compiled_ssm)
    else:
        compile_error = trial_compile_model_spec(candidate, causal_spec)
        if compile_error:
            return AssemblyValidation(
                normalized_model_spec=candidate,
                compile_ok=False,
                compile_error=str(compile_error),
                diagnostics=_collect_compile_failure_diagnostics(compile_error),
            )
        compiled_ssm = None
        compile_diagnostics = []

    if authored_priors and data_for_model is not None and not skip_ppc:
        from nof1_causal_lab.models.prior_predictive import validate_prior_predictive

        is_valid, results, raw_samples = validate_prior_predictive(
            candidate,
            authored_priors,
            data_for_model,
            data_stats=_indicator_audit_scale_stats(indicator_audits),
            causal_spec=causal_spec,
            compiled_ssm=compiled_ssm,
        )
        validation = AssemblyValidation(
            normalized_model_spec=candidate,
            compiled_ssm=compiled_ssm,
            pp_checked=True,
            pp_valid=is_valid,
            diagnostics=[*compile_diagnostics, *results],
            pp_raw_samples=raw_samples,
        )
        _attach_output_sensitivity_validation(
            validation,
            compiled_ssm=compiled_ssm,
            data_for_model=data_for_model,
        )
        return validation

    return AssemblyValidation(
        normalized_model_spec=candidate,
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

        legacy_dict = getattr(candidate, "dict", None)
        if callable(legacy_dict):
            pending.append(legacy_dict())
            continue

        if isinstance(candidate, (list, tuple, set, frozenset)):
            pending.extend(candidate)
            continue

        for attr_name in ("compile_diagnostics", "diagnostics", "errors", "results"):
            attr_value = getattr(candidate, attr_name, None)
            if attr_value is not None:
                pending.append(attr_value)

    return typed


def _attach_output_sensitivity_validation(
    validation: AssemblyValidation,
    *,
    compiled_ssm: dict[str, Any] | None,
    data_for_model: pl.DataFrame | None,
) -> None:
    """Consult Jacobian sensitivity after compile + PPC succeed."""
    if (
        compiled_ssm is None
        or data_for_model is None
        or not validation.compile_ok
        or not validation.pp_checked
        or not validation.pp_valid
    ):
        return

    consulted, supported, valid, payload, warnings = run_output_sensitivity_validation(
        compiled_ssm=compiled_ssm,
        data_for_model=data_for_model,
    )
    validation.sensitivity_consulted = consulted
    validation.sensitivity_supported = supported
    validation.sensitivity_valid = valid
    validation.sensitivity_payload = payload
    validation.sensitivity_warnings = warnings


def run_output_sensitivity_validation(
    *,
    compiled_ssm: dict[str, Any],
    data_for_model: pl.DataFrame,
) -> tuple[bool, bool, bool, dict[str, Any] | None, list[str]]:
    """Run the Stage 4 Jacobian sensitivity gate on the compiled accepted model."""
    from nof1_causal_lab.models.ssm.diagnostics import (
        OutputSensitivityUnsupportedError,
        get_stage4b_sweep_context,
        output_sensitivity_analysis,
    )
    from nof1_causal_lab.models.ssm_builder import prepare_model_runtime

    try:
        runtime = prepare_model_runtime(data_for_model=data_for_model, compiled_ssm=compiled_ssm)
        sa_result = output_sensitivity_analysis(
            runtime.model,
            runtime.times,
            observations=runtime.observations,
            n_draws=8,
            seed=42,
            sweep_context=get_stage4b_sweep_context(runtime.model),
        )
        payload = {
            "singular_values": sa_result.singular_values,
            "normalized_singular_values": sa_result.normalized_singular_values,
            "deficiency_count": sa_result.deficiency_count,
            "weak_directions": sa_result.weak_directions,
            "per_parameter": sa_result.per_parameter,
            "n_draws": sa_result.n_draws,
            "n_observations": sa_result.n_observations,
            "n_parameters": sa_result.n_parameters,
        }
        warnings = _collect_sensitivity_warning_messages(payload)
        valid = not blocking_sensitivity_fails(payload)
        return True, True, valid, payload, warnings
    except OutputSensitivityUnsupportedError as exc:
        logger.info("Stage 4 Jacobian sensitivity unavailable for this model: %s", exc)
        return True, False, True, None, [f"Jacobian sensitivity unavailable: {exc}"]
    except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as exc:
        logger.warning("Stage 4 Jacobian sensitivity failed, continuing: %s", exc)
        return True, False, True, None, [f"Jacobian sensitivity failed: {exc}"]


def _prepare_model_spec(model_spec: dict) -> dict[str, Any]:
    """Normalize a Stage 4 model spec before any compile-time work."""
    return deepcopy(model_spec)


def _collect_compile_diagnostics(compiled_ssm: dict[str, Any]) -> list[PriorValidationResult]:
    """Collect typed compiler-owned diagnostics for Stage 4 feedback."""
    from nof1_causal_lab.workers.schemas_prior import PriorValidationResult

    diagnostics = compiled_ssm.get("compile_diagnostics") or []
    typed: list[PriorValidationResult] = []
    for diagnostic in diagnostics:
        if isinstance(diagnostic, PriorValidationResult):
            typed.append(diagnostic)
        else:
            typed.append(PriorValidationResult.model_validate(diagnostic))
    return typed


def _indicator_audit_scale_stats(
    indicator_audits: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, float | None]] | None:
    """Extract the minimal per-indicator scale stats needed by prior predictive checks."""
    if not indicator_audits:
        return None

    stats: dict[str, dict[str, float | None]] = {}
    for name, audit in indicator_audits.items():
        profile = (audit or {}).get("profile") or {}
        if not profile:
            continue
        stats[name] = {
            "mean": profile.get("mean"),
            "std": profile.get("std"),
            "min": profile.get("min"),
            "max": profile.get("max"),
        }
    return stats or None


def merge_priors(existing: dict[str, dict] | None, new: dict[str, dict] | None) -> dict[str, dict]:
    """Merge prior updates into the current Stage 4 state."""
    return {**(existing or {}), **(new or {})}


def partition_prior_proposals(
    priors: dict[str, dict] | None,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Split prior proposals into schema-valid payloads and per-prior errors."""
    from nof1_causal_lab.workers.schemas_prior import PriorProposal

    validated: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for name, prior in (priors or {}).items():
        try:
            validated[name] = PriorProposal.model_validate(prior).model_dump(mode="json")
        except ValidationError as exc:
            errors[name] = str(exc)
    return validated, errors


def format_prior_proposal_errors(errors: dict[str, str]) -> str:
    """Render prior schema errors in the user-facing stage-4 format."""
    blocks = []
    for name, error in errors.items():
        lines = [f"SCHEMA ERRORS for prior '{name}':", f"- {error}"]
        if "sources." in error:
            lines.extend(
                [
                    "- `sources` must be a list of objects, not raw strings",
                    "- each source object must include `title` and `snippet`",
                    "- valid optional keys are `url`, `effect_size`, and `study_interval_days`",
                    '- example: {"title": "...", "snippet": "...", "url": "https://...", "effect_size": "β=0.2", "study_interval_days": 7.0}',
                    '- if you are unsure, use `"sources": []`',
                ]
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def validate_prior_proposals(priors: dict[str, dict] | None) -> dict[str, dict]:
    """Schema-validate prior proposals before Stage 4 assembly."""
    validated, errors = partition_prior_proposals(priors)
    if errors:
        raise ValueError(format_prior_proposal_errors(errors))
    return validated


def coerce_stage4_override_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept a replay payload and keep only authored stage-4 fields."""
    model_spec = payload.get("model_spec")
    if not isinstance(model_spec, dict):
        raise ValueError("Stage 4 replay requires a 'model_spec' object")

    authored_priors = payload.get("authored_priors")
    if not isinstance(authored_priors, dict):
        raise ValueError("Stage 4 replay requires an 'authored_priors' object")

    return {
        "model_spec": model_spec,
        "authored_priors": validate_prior_proposals(authored_priors),
        "llm_trace": payload.get("llm_trace"),
    }


def build_prior_predictive_samples(
    validation: AssemblyValidation,
    model_spec: dict,
) -> dict[str, list[float]]:
    """Forward-simulate per-variable prior predictive samples for the web payload."""
    if not validation.pp_valid or not validation.pp_raw_samples:
        return {}

    try:
        import numpy as np

        from nof1_causal_lab.artifacts.model_spec import ModelSpec

        spec = ModelSpec.model_validate(model_spec) if isinstance(model_spec, dict) else model_spec
        manifest_names = [lik.variable for lik in spec.likelihoods]
        if "observations" in validation.pp_raw_samples:
            y_np = np.asarray(validation.pp_raw_samples["observations"])
        else:
            return {}
        observation_mask = validation.pp_raw_samples.get("observations_mask")
        mask_np = np.asarray(observation_mask, dtype=bool) if observation_mask is not None else None

        samples: dict[str, list[float]] = {}
        for idx, name in enumerate(manifest_names):
            col = y_np[:, :, idx]
            if mask_np is not None and mask_np.shape == y_np.shape:
                col = col[mask_np[:, :, idx]]
            else:
                col = col[np.isfinite(col)]
            samples[name] = col.tolist()
        return samples
    except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as exc:
        raise RuntimeError(f"Prior predictive simulation failed: {exc}") from exc


def _safe_build_pp_samples(
    validation: AssemblyValidation,
    model_spec: dict,
) -> dict[str, list[float]]:
    """Build prior predictive samples, returning empty dict on failure."""
    try:
        return build_prior_predictive_samples(validation, model_spec)
    except RuntimeError:
        logger.warning("Prior predictive samples unavailable for web payload", exc_info=True)
        return {}


def build_validation_payload(
    validation: AssemblyValidation,
    model_spec: dict,
) -> dict[str, Any]:
    """Convert ``AssemblyValidation`` into the web-facing validation payload."""
    from nof1_causal_lab.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

    payload_spec = validation.normalized_model_spec or model_spec
    if not validation.compile_ok:
        return {
            "is_valid": False,
            "results": [],
            "issues": [f"Compile error: {validation.compile_error}"],
            "warnings": [],
            "prior_predictive_samples": {},
        }

    all_results = [result.model_dump() for result in validation.diagnostics]
    global_results = [
        result
        for result in validation.prior_predictive_diagnostics
        if not result.is_valid and result.parameter in GLOBAL_FAILURE_SITES
    ]
    warnings = _collect_validation_warning_messages(validation)
    if validation.has_sensitivity_failure:
        return {
            "is_valid": False,
            "results": all_results,
            "issues": [_format_sensitivity_failure_feedback(validation)],
            "warnings": warnings,
            "prior_predictive_samples": _safe_build_pp_samples(validation, payload_spec),
        }
    return {
        "is_valid": validation.is_valid,
        "results": all_results,
        "issues": (
            [_format_global_failure_summary(global_results)]
            if global_results
            else [
                result.issue
                for result in validation.prior_predictive_diagnostics
                if not result.is_valid and result.issue
            ]
        ),
        "warnings": warnings,
        "prior_predictive_samples": _safe_build_pp_samples(validation, payload_spec),
    }


def _summarize_issue_text(issue: str | None, *, max_chars: int = 240) -> str:
    """Reduce verbose exception text to the actionable root cause."""
    if not issue:
        return "Unknown validation issue"

    lines = [line.strip() for line in issue.splitlines() if line.strip()]
    if not lines:
        return "Unknown validation issue"

    summary = lines[0]
    for prefix in ("Model build failed:", "Prior predictive sampling failed:"):
        if summary.startswith(prefix):
            summary = summary.removeprefix(prefix).strip()
    if not summary and len(lines) > 1:
        summary = lines[1]
        for prefix in ("Model build failed:", "Prior predictive sampling failed:"):
            if summary.startswith(prefix):
                summary = summary.removeprefix(prefix).strip()

    bullet_lines = [
        line.lstrip("- ").strip() for line in lines[1:] if line.lstrip().startswith("-")
    ]
    if bullet_lines:
        summary = f"{summary} {bullet_lines[0]}"
        if len(bullet_lines) > 1:
            summary += f" (+{len(bullet_lines) - 1} more)"

    summary = " ".join(summary.split())
    if len(summary) <= max_chars:
        return summary

    truncated = summary[: max_chars - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated + "..."


def _format_global_failure_summary(results: list) -> str:
    """Format a concise summary for global validation failures.

    Produces a single block instead of repeating the same error for every
    parameter.  Also classifies whether the root cause is a model_spec issue
    (e.g. likelihood family incompatible with data) vs a prior issue.
    """
    lines = ["Validation FAILED (global issue — affects all parameters):"]
    seen_issues: set[str] = set()
    for r in results:
        summarized_issue = _summarize_issue_text(r.issue)
        if summarized_issue in seen_issues:
            continue
        seen_issues.add(summarized_issue)
        lines.append(f"- {summarized_issue}")
        if r.suggested_adjustment:
            lines.append(f"  Suggested: {r.suggested_adjustment}")
        if r.related_parameters:
            related = ", ".join(f"`{name}`" for name in r.related_parameters[:4])
            if len(r.related_parameters) > 4:
                related += f", +{len(r.related_parameters) - 4} more"
            lines.append(f"  Related parameters: {related}")
        if r.failure_stage:
            lines.append(f"  First failing stage: `{r.failure_stage}`")
        if r.bad_manifest_names:
            manifests = ", ".join(f"`{name}`" for name in r.bad_manifest_names[:4])
            if len(r.bad_manifest_names) > 4:
                manifests += f", +{len(r.bad_manifest_names) - 4} more"
            lines.append(f"  Bad manifests: {manifests}")
        if r.first_bad_time_index is not None:
            lines.append(f"  First bad time index: `{r.first_bad_time_index}`")
        if r.supporting_codes:
            codes = ", ".join(f"`{code}`" for code in r.supporting_codes[:4])
            if len(r.supporting_codes) > 4:
                codes += f", +{len(r.supporting_codes) - 4} more"
            lines.append(f"  Supporting diagnostics: {codes}")

    # Heuristic: observation-support errors are model_spec issues, not prior issues.
    issues_text = " ".join(r.issue or "" for r in results)
    if "support check" in issues_text.lower() or "outside support" in issues_text.lower():
        lines.append("")
        lines.append(
            "NOTE: This is a model_spec issue (likelihood family incompatible "
            "with observed data). Consider changing the distribution family "
            "rather than adjusting priors."
        )

    return "\n".join(lines)


def _collect_validation_warning_messages(validation: AssemblyValidation) -> list[str]:
    """Flatten warning diagnostics into user-facing text."""
    messages = [
        result.issue
        for result in validation.diagnostics
        if result.severity == "warning" and result.issue
    ]
    messages.extend(validation.sensitivity_warnings)
    return [message for message in messages if isinstance(message, str)]


# Shared headers and remediations for warning codes that repeat across edges.
# When multiple warnings share a code, the renderer emits the header and
# suggested-adjustment text once and lists per-edge facts as bullets, which
# saves ~100 tokens per extra warning on dt_ct_approximation_warning alone.
_WARNING_GROUP_TEMPLATES: dict[str, tuple[str, str]] = {
    "dt_ct_approximation_warning": (
        "Cross-lag diagnostics evaluate the full matrix logarithm "
        "`logm(A_dt) / dt`. The elementwise `beta_dt / dt` CT coupling differs "
        "materially from that full-system matrix-log scale for these edges:",
        "Use the exact matrix-log CT scale when revising these edges: shorten the "
        "reference interval, shrink the DT beta prior, or elicit the prior directly "
        "on the CT rate.",
    ),
    "lagged_response_weak": (
        "Across prior draws, the full-system one-lag response is near-zero "
        "on the declared lag for some edges:",
        "Confirm that a near-zero one-lag effect is substantively intended. "
        "If not, strengthen the daily-scale prior or author it on the source "
        "study interval with `reference_interval_days`.",
    ),
    "interval_reference_missing": (
        "`reference_interval_days` is omitted, so the prior is being interpreted "
        "on the default model interval even though the cited evidence is on a "
        "different study interval:",
        "If the authored effect is meant to be on the source study interval, "
        "set `reference_interval_days` to that interval. Otherwise explain why "
        "a daily-scale prior is appropriate.",
    ),
    "interval_reference_mismatch": (
        "The authored `reference_interval_days` disagrees materially with the "
        "cited study interval:",
        "Confirm that the prior was intentionally rescaled to the authored "
        "interval, or align `reference_interval_days` with the evidence interval.",
    ),
    "interval_sources_mixed": (
        "Cited sources mix materially different study intervals, so the "
        "authored interval provenance is weak:",
        "Use sources measured on a comparable interval when possible, or "
        "explain which interval the prior is expressed on with "
        "`reference_interval_days`.",
    ),
}


def _format_validation_warnings(validation: AssemblyValidation) -> str:
    """Render non-fatal validation diagnostics.

    Warnings sharing a ``code`` are collapsed into one block: the shared
    explanation and remediation are emitted once, and each edge contributes
    one fact-only bullet underneath. Warnings without a registered template
    fall back to the per-warning rendering.
    """
    grouped_warnings: list[Any] = list(validation.compile_diagnostics)
    grouped_warnings.extend(
        result for result in validation.prior_predictive_diagnostics if result.severity == "warning"
    )

    by_code: dict[str, list[Any]] = {}
    ordered_codes: list[str] = []
    for warning in grouped_warnings:
        code = warning.code or ""
        if code not in by_code:
            by_code[code] = []
            ordered_codes.append(code)
        by_code[code].append(warning)

    parts: list[str] = []
    for code in ordered_codes:
        group = by_code[code]
        template = _WARNING_GROUP_TEMPLATES.get(code)
        if template is not None:
            header, suggested = template
            block_lines = [header]
            for warning in group:
                if warning.issue:
                    block_lines.append(f"- {warning.issue}")
            if len(block_lines) == 1:
                continue
            if suggested:
                block_lines.append(f"  Suggested: {suggested}")
            parts.append("\n".join(block_lines))
            continue

        for warning in group:
            if not warning.issue:
                continue
            lines = [f"- {warning.issue}"]
            if warning.suggested_adjustment:
                lines.append(f"  Suggested: {warning.suggested_adjustment}")
            parts.append("\n".join(lines))

    for warning in validation.sensitivity_warnings:
        if warning:
            parts.append(f"- {warning}")

    if not parts:
        return ""
    return "MODELING WARNINGS:\n" + "\n\n".join(parts)


_TAU_SITE_NAMES = frozenset({"static_state_sd", "static_state_sd_free"})
_TAU_DOMINANCE_THRESHOLD = 0.9


def _loading_is_tau_family(loading: dict[str, Any]) -> bool:
    """Return True when a sensitivity loading targets a static-state SD (tau)."""
    parameter = loading.get("parameter")
    if isinstance(parameter, str):
        site = parameter.split("[", 1)[0]
        if site in _TAU_SITE_NAMES:
            return True
    interpretable = loading.get("interpretable_parameter")
    return isinstance(interpretable, str) and interpretable.startswith("tau_")


def _direction_is_tau_dominated(direction: dict[str, Any]) -> bool:
    """Return True when a fail direction's squared loadings concentrate on tau.

    Static-state SDs are structurally weakly identified in N-of-1 settings —
    the prior carries them and the agent cannot repair them by re-eliciting.
    Fail directions dominated by taus surface as warnings rather than blocking
    acceptance.
    """
    loadings = [entry for entry in direction.get("top_loadings", []) if isinstance(entry, dict)]
    if not loadings:
        return False
    total_sq = 0.0
    tau_sq = 0.0
    for loading in loadings:
        try:
            value = float(loading.get("loading", 0.0))
        except (TypeError, ValueError):
            continue
        sq = value * value
        total_sq += sq
        if _loading_is_tau_family(loading):
            tau_sq += sq
    if total_sq <= 0.0:
        return False
    return tau_sq / total_sq >= _TAU_DOMINANCE_THRESHOLD


def blocking_sensitivity_fails(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return fail directions that should block Stage 4 acceptance.

    Tau-dominated fail directions are demoted to warnings; only mixed or
    structural-parameter fails block.
    """
    if not payload:
        return []
    return [
        direction
        for direction in payload.get("weak_directions", [])
        if isinstance(direction, dict)
        and direction.get("status") == "fail"
        and not _direction_is_tau_dominated(direction)
    ]


def _demoted_sensitivity_fails(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return fail directions demoted to warnings because they are tau-dominated."""
    if not payload:
        return []
    return [
        direction
        for direction in payload.get("weak_directions", [])
        if isinstance(direction, dict)
        and direction.get("status") == "fail"
        and _direction_is_tau_dominated(direction)
    ]


def _collect_sensitivity_warning_messages(
    payload: dict[str, Any] | None,
) -> list[str]:
    """Render non-fatal Jacobian-sensitivity warnings for accepted state."""
    if not payload:
        return []

    warnings: list[str] = []
    weak_directions = [
        direction
        for direction in payload.get("weak_directions", [])
        if isinstance(direction, dict) and direction.get("status") == "warn"
    ]
    for direction in weak_directions[:2]:
        warnings.append(_format_sensitivity_direction_message(direction, prefix="Warning"))
    for direction in _demoted_sensitivity_fails(payload)[:2]:
        warnings.append(
            _format_sensitivity_direction_message(
                direction, prefix="Warning (tau-dominated, unidentifiable by design)"
            )
        )
    return warnings


def _format_sensitivity_direction_message(
    direction: dict[str, Any],
    *,
    prefix: str,
) -> str:
    """Render one weak normalized sensitivity direction into compact text.

    Includes signed loadings so the agent sees both the relative
    contribution of each parameter (magnitude) and the sign pattern
    (which combination is locally unidentified — e.g. ``+a, +b`` vs
    ``+a, -b``). Loadings are listed in descending absolute magnitude
    and truncated when the running absolute coverage exceeds 0.95 or
    after 8 terms, whichever comes first.
    """
    index = direction.get("index")
    normalized_sv = direction.get("normalized_singular_value")
    try:
        normalized_sv_text = f"{float(normalized_sv):.3g}"
    except (TypeError, ValueError):
        normalized_sv_text = "unknown"
    loadings = [
        loading for loading in direction.get("top_loadings", []) if isinstance(loading, dict)
    ]
    loadings.sort(key=lambda item: float(item.get("abs_loading") or 0.0), reverse=True)
    rendered: list[str] = []
    cumulative_sq = 0.0
    for loading in loadings:
        name = str(loading.get("interpretable_parameter") or loading.get("parameter") or "")
        if not name:
            continue
        try:
            signed = float(loading.get("loading") or 0.0)
        except (TypeError, ValueError):
            continue
        rendered.append(f"{name}={signed:+.2f}")
        cumulative_sq += signed * signed
        if len(rendered) >= 8 or cumulative_sq >= 0.95:
            break
    loadings_text = ", ".join(rendered) if rendered else "the active parameter surface"
    return (
        f"{prefix}: Jacobian sensitivity found weak normalized direction "
        f"{index} (normalized singular value={normalized_sv_text}); "
        f"top signed loadings: {loadings_text}."
    )


def _format_sensitivity_failure_feedback(validation: AssemblyValidation) -> str:
    """Format every failing Jacobian-sensitivity direction for Stage 4 feedback.

    The agent needs to see all fail directions (not just the worst one)
    because two unrelated unidentified combinations can exist in the same
    model and fixing only the top direction leaves the second failing on
    the next round. Each direction is rendered with its signed loadings
    so the agent can reason about the specific coupled combination.
    """
    payload = validation.sensitivity_payload or {}
    fail_directions = blocking_sensitivity_fails(payload)
    if not fail_directions:
        return "JACOBIAN SENSITIVITY FEEDBACK:\n- the current accepted model remains locally weak"

    fail_directions.sort(
        key=lambda item: (
            float(item.get("normalized_singular_value", float("inf"))),
            int(item.get("index", 0)),
        ),
    )
    lines = ["JACOBIAN SENSITIVITY FEEDBACK:"]
    for direction in fail_directions:
        message = _format_sensitivity_direction_message(direction, prefix="Failure")
        lines.append(f"- {message.removeprefix('Failure: ')}")
    lines.append(
        "- the accepted parameterization is still locally weak along "
        f"{'this coupled direction' if len(fail_directions) == 1 else 'these coupled directions'}"
    )
    lines.append(
        "- loadings are the normalized right-singular vector entries; sign "
        "indicates the combination (co-increase vs anti-correlate) that data "
        "cannot distinguish at the current priors"
    )
    return "\n".join(lines)


def compile_model_artifact(
    model_spec: dict,
    authored_priors: dict[str, dict],
    data_for_model: pl.DataFrame,
    causal_spec: dict | None = None,
    compiled_ssm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile and verify the executable SSM artifact for Stage 4 output."""
    from nof1_causal_lab.models.ssm_builder import prepare_model_runtime
    from nof1_causal_lab.models.ssm_compiler import compile_ssm_artifact

    try:
        artifact = compiled_ssm or compile_ssm_artifact(
            _prepare_model_spec(model_spec),
            authored_priors,
            causal_spec=causal_spec,
        )
    except _RECOVERABLE_STAGE4_ASSEMBLY_ERRORS as exc:
        return {
            "model_built": False,
            "error": str(exc),
        }

    try:
        runtime = prepare_model_runtime(data_for_model, compiled_ssm=artifact)
        builder = runtime.builder
        return {
            "model_built": True,
            "model_type": builder.model_type,
            "version": builder.version,
            "compiled_ssm": artifact,
        }
    except NotImplementedError:
        return {
            "model_built": False,
            "error": "SSM implementation not available",
            "compiled_ssm": artifact,
        }
    except _RECOVERABLE_STAGE4_ASSEMBLY_ERRORS as exc:
        return {
            "model_built": False,
            "error": str(exc),
            "compiled_ssm": artifact,
        }


def materialize_stage4_result(
    *,
    model_spec: dict[str, Any],
    authored_priors: dict[str, dict],
    data_for_model: pl.DataFrame,
    indicator_audits: dict[str, dict[str, Any]] | None,
    causal_spec: dict | None,
    llm_trace: dict[str, Any] | None = None,
    validation: AssemblyValidation | None = None,
    search_queries: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the full grounded stage-4 result from authored inputs."""
    from nof1_causal_lab.models.ssm_compiler import resolve_prior_proposals

    validation = validation or validate_assembly(
        model_spec,
        authored_priors,
        data_for_model,
        indicator_audits,
        causal_spec,
    )
    normalized_model_spec = validation.normalized_model_spec or model_spec
    validation_result = build_validation_payload(validation, normalized_model_spec)
    model_result = compile_model_artifact(
        normalized_model_spec,
        authored_priors,
        data_for_model,
        causal_spec=causal_spec,
        compiled_ssm=validation.compiled_ssm,
    )
    compiled_ssm = model_result.pop("compiled_ssm", None)
    resolved_priors = (
        resolve_prior_proposals(compiled_ssm, authored_priors=authored_priors)
        if compiled_ssm
        else []
    )

    result = {
        "model_spec": normalized_model_spec,
        "authored_priors": authored_priors,
        "resolved_priors": resolved_priors,
        "search_queries": search_queries or None,
        "validation_warnings": validation_result.get("warnings") or None,
        "_causal_spec": causal_spec,
        "prior_predictive_samples": validation_result.get("prior_predictive_samples", {}),
    }
    if compiled_ssm is not None:
        result["_compiled_ssm"] = compiled_ssm
    if llm_trace is not None:
        result["llm_trace"] = llm_trace
    return result


def format_validation_feedback(
    validation: AssemblyValidation,
    authored_priors: dict,
    *,
    focus_parameters: Iterable[str] | None = None,
    data_stats: dict | None = None,
) -> str:
    """Format assembly validation result as feedback string.

    ``focus_parameters`` narrows which failing parameters get rendered as
    detailed feedback — intended for state-machine callers that want to
    show the LLM only what belongs to its active block. When ``None`` the
    full set of failing (non-global) parameters is rendered, which is the
    right behavior for scope-free callers such as the megaprompt path and
    the grounding pipeline itself. Global failures (log-link overflow,
    stability, etc.) are always surfaced regardless of the focus set so
    the LLM can never be silently driven into an unfixable corner.
    """
    if not validation.compile_ok:
        return f"COMPILE ERROR:\n{validation.compile_error}"

    warning_feedback = _format_validation_warnings(validation)
    if validation.has_sensitivity_failure:
        details = _format_sensitivity_failure_feedback(validation)
        if warning_feedback:
            details = f"{details}\n\n{warning_feedback}"
        return details

    if not validation.pp_checked or validation.pp_valid:
        return warning_feedback or "VALID"

    from nof1_causal_lab.models.prior_predictive import format_parameter_feedback
    from nof1_causal_lab.models.ssm_compilation_common import GLOBAL_FAILURE_SITES

    # Global failures → single concise summary, not one block per parameter.
    # Always shown regardless of ``focus_parameters`` — they affect the whole
    # system.
    global_results = [
        r
        for r in validation.prior_predictive_diagnostics
        if not r.is_valid and r.parameter in GLOBAL_FAILURE_SITES
    ]
    if global_results:
        details = _format_global_failure_summary(global_results)
        if warning_feedback:
            details = f"{details}\n\n{warning_feedback}"
        return f"PRIOR PREDICTIVE FEEDBACK:\n{details}"

    # Decide which per-parameter failures to render. Callers that want a
    # scoped view (state-machine reducer for an active block or repair
    # campaign) pass ``focus_parameters`` explicitly. Scope-free callers
    # leave it as ``None`` and get every failing parameter the validator
    # flagged.
    if focus_parameters is None:
        seen: set[str] = set()
        failing_param_names: list[str] = []
        for result in validation.prior_predictive_diagnostics:
            if result.is_valid or result.parameter in GLOBAL_FAILURE_SITES:
                continue
            if result.parameter in seen:
                continue
            seen.add(result.parameter)
            failing_param_names.append(result.parameter)
        params: list[str] = failing_param_names
    else:
        params = list(focus_parameters)

    parts: list[str] = []
    for param_name in params:
        fb = format_parameter_feedback(
            parameter_name=param_name,
            results=validation.prior_predictive_diagnostics,
            prior=authored_priors.get(param_name),
            data_stats=data_stats,
            model_spec=validation.normalized_model_spec,
        )
        if fb:
            parts.append(fb)

    details = "\n\n".join(parts) if parts else "PRIOR PREDICTIVE CHECK FAILED"
    if warning_feedback:
        details = f"{details}\n\n{warning_feedback}"
    return f"PRIOR PREDICTIVE FEEDBACK:\n{details}"
