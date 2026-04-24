"""Stage 3 rule registry and reduction logic."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.flows.stages.stage3.checks import (
    MIN_OBSERVATIONS,
    OBSERVATION_TIME_COLUMN,
    check_construct_correlations,
    check_dtype_range,
    check_hallucination_signals,
    check_time_coverage,
    check_timestamp_gaps,
    parse_timestamp_series,
    timestamp_issue_specs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_prefect_logger(__name__)


@dataclass(frozen=True)
class Issue:
    indicator: str
    issue_type: str
    severity: str
    message: str
    cell_key: str


@dataclass
class ValidationFindings:
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorContext:
    name: str
    ind_data: pl.DataFrame
    values: pl.Series
    n_obs: int
    variance: float | None
    dtype: str | None
    is_time_invariant: bool
    model_clock_hours: float | None
    parsed_ts: pl.Series
    n_unparseable: int
    n_total_ts: int


@dataclass(frozen=True)
class IndicatorRuleInput:
    name: str
    ind_data: pl.DataFrame
    ctx: IndicatorContext | None


@dataclass(frozen=True)
class ValidationContext:
    combined: pl.DataFrame
    indicators: list[dict]
    indicator_names: set[str]
    indicator_lookup: dict[str, dict]
    construct_lookup: dict[str, dict]
    model_clock_hours: float | None

    def iter_indicators(self):
        for indicator_name in sorted(self.indicator_names):
            ind_data = self.combined.filter(pl.col("indicator") == indicator_name)
            if ind_data.is_empty():
                yield indicator_name, ind_data, None
                continue
            yield (
                indicator_name,
                ind_data,
                _build_indicator_context(
                    indicator_name,
                    ind_data,
                    self.indicator_lookup,
                    self.construct_lookup,
                    self.model_clock_hours,
                ),
            )


@dataclass(frozen=True)
class ValidationRule:
    name: str
    scope: str
    check: Any


def issue_payload(issue: Issue) -> dict[str, str | None]:
    return {
        "indicator": issue.indicator,
        "issue_type": issue.issue_type,
        "severity": issue.severity,
        "message": issue.message,
    }


def issue_from_raw(raw_issue: dict[str, Any], *, cell_key: str) -> Issue:
    return Issue(
        raw_issue["indicator"],
        raw_issue["issue_type"],
        raw_issue["severity"],
        raw_issue["message"],
        cell_key=cell_key,
    )


def issues_from_raw(
    raw_issues: list[dict[str, Any]],
    *,
    cell_key: str | Callable[[dict[str, Any]], str],
) -> list[Issue]:
    issues: list[Issue] = []
    for raw_issue in raw_issues:
        resolved_cell_key = cell_key if isinstance(cell_key, str) else cell_key(raw_issue)
        issues.append(issue_from_raw(raw_issue, cell_key=resolved_cell_key))
    return issues


def derive_validation_status(
    issues: list[dict[str, Any]],
) -> dict[str, bool | Literal["success", "warn", "fail"] | str | None]:
    has_error = any(issue.get("severity") == "error" for issue in issues)
    has_warning = any(issue.get("severity") == "warning" for issue in issues)

    if has_error:
        return {
            "is_valid": False,
            "outcome": "fail",
            "fail_reason": "data_validation_failed",
        }
    if has_warning:
        return {
            "is_valid": True,
            "outcome": "warn",
            "fail_reason": None,
        }
    return {
        "is_valid": True,
        "outcome": "success",
        "fail_reason": None,
    }


def no_data_validation_result() -> dict[str, Any]:
    return {
        "is_valid": False,
        "indicators": {},
        "dataset_issues": [
            {
                "indicator": None,
                "issue_type": "no_data",
                "severity": "error",
                "message": "No data extracted",
            }
        ],
    }


def _rule_missing(entry: IndicatorRuleInput) -> ValidationFindings:
    if not entry.ind_data.is_empty():
        return ValidationFindings()
    return ValidationFindings(
        issues=[
            Issue(
                entry.name,
                "missing",
                "warning",
                "No data extracted for this indicator",
                cell_key="",
            )
        ]
    )


def _rule_no_numeric(entry: IndicatorRuleInput) -> ValidationFindings:
    if entry.ind_data.is_empty() or entry.ctx is not None:
        return ValidationFindings()
    return ValidationFindings(
        issues=[
            Issue(
                entry.name,
                "no_numeric",
                "error",
                "No numeric values extracted",
                cell_key="",
            )
        ]
    )


def _rule_timestamps(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    issues = [
        Issue(
            ctx.name,
            "unparseable_timestamps",
            severity,
            message,
            cell_key="n_unparseable_timestamps",
        )
        for severity, message in timestamp_issue_specs(ctx.n_total_ts, ctx.n_unparseable)
    ]
    return ValidationFindings(
        issues=issues,
        metrics={"n_unparseable_timestamps": ctx.n_unparseable},
    )


def _rule_sample_size(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    issues = []
    if ctx.n_obs < MIN_OBSERVATIONS:
        issues.append(
            Issue(
                ctx.name,
                "low_n",
                "warning",
                f"Only {ctx.n_obs} observations (recommend >= {MIN_OBSERVATIONS})",
                cell_key="n_obs",
            )
        )
    return ValidationFindings(issues=issues, metrics={"n_obs": ctx.n_obs})


def _rule_variance(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    issues = []
    if ctx.variance is not None and ctx.variance == 0:
        issues.append(
            Issue(
                ctx.name,
                "no_variance",
                "error",
                f"Zero variance (constant value = {ctx.values.first()})",
                cell_key="variance",
            )
        )
    return ValidationFindings(issues=issues, metrics={"variance": ctx.variance})


def _rule_dtype_range(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    if not ctx.dtype:
        return ValidationFindings(metrics={"dtype_violations": 0})
    raw_issues, violation_count = check_dtype_range(ctx.values, ctx.dtype, ctx.name)
    return ValidationFindings(
        issues=issues_from_raw(raw_issues, cell_key="dtype_violations"),
        metrics={"dtype_violations": violation_count},
    )


def _rule_time_coverage(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    if ctx.is_time_invariant or ctx.model_clock_hours is None:
        return ValidationFindings(metrics={"time_coverage_ratio": None})
    raw_issues, ratio = check_time_coverage(ctx.parsed_ts, ctx.model_clock_hours, ctx.name)
    return ValidationFindings(
        issues=issues_from_raw(raw_issues, cell_key="time_coverage_ratio"),
        metrics={"time_coverage_ratio": ratio},
    )


def _rule_timestamp_gaps(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    if ctx.is_time_invariant or ctx.model_clock_hours is None:
        return ValidationFindings(metrics={"max_gap_ratio": None})
    raw_issues, ratio = check_timestamp_gaps(ctx.parsed_ts, ctx.model_clock_hours, ctx.name)
    return ValidationFindings(
        issues=issues_from_raw(raw_issues, cell_key="max_gap_ratio"),
        metrics={"max_gap_ratio": ratio},
    )


def _rule_hallucination_signals(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    raw_issues, duplicate_pct, arith_seq = check_hallucination_signals(
        ctx.values, ctx.dtype or "continuous", ctx.name
    )
    return ValidationFindings(
        issues=issues_from_raw(
            raw_issues,
            cell_key=lambda raw_issue: (
                "arithmetic_sequence_detected"
                if "arithmetic sequence" in raw_issue["message"]
                else "duplicate_pct"
            ),
        ),
        metrics={"duplicate_pct": duplicate_pct, "arithmetic_sequence_detected": arith_seq},
    )


def _rule_construct_correlations(
    combined: pl.DataFrame, indicators: list[dict]
) -> ValidationFindings:
    raw_issues = check_construct_correlations(combined, indicators)
    return ValidationFindings(issues=issues_from_raw(raw_issues, cell_key=""))


RULES: list[ValidationRule] = [
    ValidationRule("missing", "indicator", _rule_missing),
    ValidationRule("no_numeric", "indicator", _rule_no_numeric),
    ValidationRule("timestamps", "indicator", _rule_timestamps),
    ValidationRule("sample_size", "indicator", _rule_sample_size),
    ValidationRule("variance", "indicator", _rule_variance),
    ValidationRule("dtype_range", "indicator", _rule_dtype_range),
    ValidationRule("time_coverage", "indicator", _rule_time_coverage),
    ValidationRule("timestamp_gaps", "indicator", _rule_timestamp_gaps),
    ValidationRule("hallucination_signals", "indicator", _rule_hallucination_signals),
    ValidationRule("construct_correlations", "dataset", _rule_construct_correlations),
]

CELL_STATUS_KEYS = frozenset(
    {
        "n_obs",
        "variance",
        "n_unparseable_timestamps",
        "time_coverage_ratio",
        "max_gap_ratio",
        "dtype_violations",
        "duplicate_pct",
        "arithmetic_sequence_detected",
    }
)


def reduce_findings(
    indicator_findings: dict[str, list[ValidationFindings]],
) -> tuple[list[dict], dict[str, dict[str, Any]]]:
    all_issues: list[dict] = []
    indicator_health: dict[str, dict[str, Any]] = {}

    for indicator_name in sorted(indicator_findings):
        findings_list = indicator_findings[indicator_name]
        merged_metrics: dict[str, Any] = {}
        ind_issues: list[Issue] = []
        for findings in findings_list:
            ind_issues.extend(findings.issues)
            merged_metrics.update(findings.metrics)

        for issue in ind_issues:
            all_issues.append(issue_payload(issue))

        cell_statuses: dict[str, str] = {
            key: "ok" for key in CELL_STATUS_KEYS if key in merged_metrics
        }
        for issue in ind_issues:
            if issue.cell_key in cell_statuses and cell_statuses[issue.cell_key] != "error":
                cell_statuses[issue.cell_key] = issue.severity

        indicator_health[indicator_name] = {
            **{key: value for key, value in merged_metrics.items() if key in CELL_STATUS_KEYS},
            "cell_statuses": cell_statuses,
        }

    return all_issues, indicator_health


def _build_indicator_context(
    indicator_name: str,
    ind_data: pl.DataFrame,
    indicator_lookup: dict[str, dict],
    construct_lookup: dict[str, dict],
    model_clock_hours: float | None,
) -> IndicatorContext | None:
    values_df = ind_data.select(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()
    n_obs = len(values_df)
    if n_obs == 0:
        return None

    values = values_df["value"]
    variance: float | None = None
    try:
        raw_variance = values.var()
        if isinstance(raw_variance, timedelta):
            variance = raw_variance.total_seconds()
        elif raw_variance is not None:
            variance = float(raw_variance)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        logger.info("Variance calculation failed for indicator %s", indicator_name, exc_info=True)

    indicator_meta = indicator_lookup.get(indicator_name, {})
    dtype = indicator_meta.get("measurement_dtype")
    construct_name = indicator_meta.get("construct_name")
    construct_meta = construct_lookup.get(construct_name, {}) if construct_name else {}
    is_time_invariant = construct_meta.get("temporal_status") == "time_invariant"

    timestamps = ind_data[OBSERVATION_TIME_COLUMN]
    parsed = parse_timestamp_series(timestamps)

    return IndicatorContext(
        name=indicator_name,
        ind_data=ind_data,
        values=values,
        n_obs=n_obs,
        variance=variance,
        dtype=dtype,
        is_time_invariant=is_time_invariant,
        model_clock_hours=model_clock_hours,
        parsed_ts=parsed.drop_nulls(),
        n_unparseable=parsed.null_count(),
        n_total_ts=len(timestamps),
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric


def _compute_empirical_profile(
    indicator_name: str,
    model_data: pl.DataFrame,
    indicator_lookup: dict[str, dict],
    health_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    ind_model = model_data.filter(pl.col("indicator") == indicator_name)
    values_df = ind_model.select(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()
    n_obs = len(values_df)
    if n_obs == 0:
        return None

    values = values_df["value"]
    mean = _float_or_none(values.mean())
    variance = _float_or_none(values.var())
    min_value = _float_or_none(values.min())
    max_value = _float_or_none(values.max())
    numeric_values = [float(v) for v in values.to_list()]

    return {
        "measurement_dtype": indicator_lookup.get(indicator_name, {}).get("measurement_dtype"),
        "n_obs": n_obs,
        "mean": mean,
        "std": _float_or_none(values.std()),
        "min": min_value,
        "max": max_value,
        "q25": _float_or_none(values.quantile(0.25)),
        "q50": _float_or_none(values.quantile(0.50)),
        "q75": _float_or_none(values.quantile(0.75)),
        "variance": variance,
        "time_coverage_ratio": _float_or_none(health_metrics.get("time_coverage_ratio")),
        "max_gap_ratio": _float_or_none(health_metrics.get("max_gap_ratio")),
        "dtype_violations": health_metrics.get("dtype_violations"),
        "duplicate_pct": _float_or_none(health_metrics.get("duplicate_pct")),
        "arithmetic_sequence_detected": bool(
            health_metrics.get("arithmetic_sequence_detected", False)
        ),
        "n_unparseable_timestamps": health_metrics.get("n_unparseable_timestamps"),
        "zero_fraction": (
            float(
                sum(1 for value in numeric_values if math.isclose(value, 0.0, abs_tol=1e-12))
                / n_obs
            )
            if n_obs > 0
            else None
        ),
        "is_nonnegative": min_value >= 0 if min_value is not None else None,
        "is_unit_interval": (
            min_value >= 0 and max_value <= 1
            if min_value is not None and max_value is not None
            else None
        ),
        "looks_integer_valued": all(
            math.isclose(value, round(value), abs_tol=1e-8) for value in numeric_values
        ),
        "variance_to_mean_ratio": (
            variance / mean if variance is not None and mean is not None and mean > 0 else None
        ),
    }


def build_indicator_audits(
    *,
    indicator_names: set[str],
    indicator_lookup: dict[str, dict],
    model_data: pl.DataFrame,
    indicator_issues: list[dict[str, Any]],
    indicator_health: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    issues_by_indicator: dict[str, list[dict[str, Any]]] = {name: [] for name in indicator_names}
    for issue in indicator_issues:
        issue_indicator = issue.get("indicator")
        if issue_indicator in issues_by_indicator:
            issues_by_indicator[issue_indicator].append(issue)

    audits: dict[str, dict[str, Any]] = {}
    for indicator_name in sorted(indicator_names):
        health_metrics = indicator_health.get(indicator_name, {})
        audits[indicator_name] = {
            "profile": _compute_empirical_profile(
                indicator_name,
                model_data,
                indicator_lookup,
                health_metrics,
            ),
            "validation": {
                "issues": issues_by_indicator.get(indicator_name, []),
                "checks": dict(health_metrics.get("cell_statuses", {})),
            },
        }
    return audits


def run_rules(
    rules: list[ValidationRule],
    ctx: ValidationContext,
) -> tuple[list[dict], dict[str, dict[str, Any]], list[dict]]:
    indicator_rules = [rule for rule in rules if rule.scope == "indicator"]
    dataset_rules = [rule for rule in rules if rule.scope == "dataset"]

    indicator_findings: dict[str, list[ValidationFindings]] = {}
    for indicator_name, ind_data, indicator_ctx in ctx.iter_indicators():
        rule_input = IndicatorRuleInput(indicator_name, ind_data, indicator_ctx)
        indicator_findings[indicator_name] = [rule.check(rule_input) for rule in indicator_rules]

    dataset_issues: list[dict] = []
    for rule in dataset_rules:
        findings = rule.check(ctx.combined, ctx.indicators)
        dataset_issues.extend(issue_payload(issue) for issue in findings.issues)

    indicator_issues, indicator_health = reduce_findings(indicator_findings)
    return indicator_issues, indicator_health, dataset_issues
