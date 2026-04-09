"""Stage 3: Validate extracted data.

Validation is expressed as composable ``ValidationRule`` entries. Each rule
returns ``ValidationFindings`` (issues with ``cell_key`` + raw metrics).
A single ``reduce_findings()`` derives cell statuses from issues, so
threshold logic lives inside rules and status derivation is centralized.

Adding a new validation check = appending one entry to ``RULES``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.flows import get_prefect_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_prefect_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

MIN_OBSERVATIONS = 10
MIN_COVERAGE_PERIODS = 10
MAX_GAP_MULTIPLIER = 5
OUTLIER_IQR_MULTIPLIER = 3.0
MIN_ALIGNED_FOR_CFA = 10
HALLUCINATION_DUPLICATE_THRESHOLD = 0.5
OBSERVATION_TIME_COLUMN = "anchor_time"

_TIMESTAMP_FORMATS: tuple[tuple[str | None, bool], ...] = (
    ("%Y-%m-%d %H:%M:%S", False),
    ("%Y-%m-%d %H:%M:%S%.f", False),
    ("%Y-%m-%d %H:%M", False),
    ("%Y-%m-%d", False),
    ("%Y-%m-%dT%H:%M:%S", False),
    ("%Y-%m-%dT%H:%M:%S%.f", False),
    ("%Y-%m-%dT%H:%M:%S%z", True),
    ("%Y-%m-%dT%H:%M:%S%.f%z", True),
    ("%Y-%m-%d %H:%M:%S%z", True),
    ("%Y-%m-%d %H:%M%z", True),
)


# ══════════════════════════════════════════════════════════════════════════════
# Core abstractions
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Issue:
    """A single validation issue linked to a metric via ``cell_key``."""

    indicator: str
    issue_type: str
    severity: str  # "warning" or "error"
    message: str
    cell_key: str  # metric this pertains to, e.g. "n_obs", "variance"


@dataclass
class ValidationFindings:
    """Output of a single rule: issues (with cell_keys) + raw metrics."""

    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorContext:
    """Pre-computed per-indicator state shared across rules."""

    name: str
    ind_data: pl.DataFrame
    values: pl.Series  # numeric Float64, nulls dropped
    n_obs: int
    variance: float | None
    dtype: str | None
    is_time_invariant: bool
    model_clock_hours: float | None
    parsed_ts: pl.Series  # parsed timestamps, nulls dropped
    n_unparseable: int
    n_total_ts: int


@dataclass(frozen=True)
class IndicatorRuleInput:
    """One indicator's raw data plus any derived numeric/timestamp context."""

    name: str
    ind_data: pl.DataFrame
    ctx: IndicatorContext | None


@dataclass(frozen=True)
class ValidationContext:
    """Dataset-level validation context with helpers for per-indicator views."""

    combined: pl.DataFrame
    indicators: list[dict]
    indicator_names: set[str]
    indicator_lookup: dict[str, dict]
    construct_lookup: dict[str, dict]
    model_clock_hours: float | None

    def iter_indicators(self):
        for ind_name in sorted(self.indicator_names):
            ind_data = self.combined.filter(pl.col("indicator") == ind_name)
            if ind_data.is_empty():
                yield ind_name, ind_data, None
                continue
            yield (
                ind_name,
                ind_data,
                _build_indicator_context(
                    ind_name,
                    ind_data,
                    self.indicator_lookup,
                    self.construct_lookup,
                    self.model_clock_hours,
                ),
            )


@dataclass(frozen=True)
class ValidationRule:
    """Composable validation rule."""

    name: str
    scope: str  # "indicator" or "dataset"
    check: Any  # (IndicatorRuleInput) -> ValidationFindings or dataset check callable


def _issue_from_raw(
    raw_issue: dict[str, Any],
    *,
    cell_key: str,
) -> Issue:
    return Issue(
        raw_issue["indicator"],
        raw_issue["issue_type"],
        raw_issue["severity"],
        raw_issue["message"],
        cell_key=cell_key,
    )


def _issues_from_raw(
    raw_issues: list[dict[str, Any]],
    *,
    cell_key: str | Callable[[dict[str, Any]], str],
) -> list[Issue]:
    """Convert raw issue dicts into Issue dataclasses with centralized cell-key mapping."""
    issues: list[Issue] = []
    for raw_issue in raw_issues:
        resolved_cell_key = cell_key if isinstance(cell_key, str) else cell_key(raw_issue)
        issues.append(_issue_from_raw(raw_issue, cell_key=resolved_cell_key))
    return issues


def _issue_payload(issue: Issue) -> dict[str, str | None]:
    return {
        "indicator": issue.indicator,
        "issue_type": issue.issue_type,
        "severity": issue.severity,
        "message": issue.message,
    }


def derive_validation_status(
    issues: list[dict[str, Any]],
) -> dict[str, bool | Literal["success", "warn", "fail"] | str | None]:
    """Derive Stage 3 validity and outcome directly from local issue severities."""
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


def _no_data_validation_result() -> dict[str, Any]:
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


# ══════════════════════════════════════════════════════════════════════════════
# Low-level check functions (preserved for direct testability)
# ══════════════════════════════════════════════════════════════════════════════


def _timestamp_issue_specs(
    n_total: int,
    n_unparseable: int,
) -> list[tuple[str, str]]:
    """Return normalized observation-time parseability issues from aggregate counts."""
    if n_total == 0:
        return []
    if n_unparseable == n_total:
        return [("error", f"All {n_total} timestamps are unparseable")]
    if n_unparseable > n_total * 0.5:
        return [("warning", f"{n_unparseable}/{n_total} timestamps are unparseable (>50%)")]
    return []


def _check_dtype_range(values: pl.Series, dtype: str, ind_name: str) -> tuple[list[dict], int]:
    """Check values conform to declared measurement dtype.

    Returns:
        Tuple of (issues, dtype_violation_count).
    """
    issues: list[dict] = []
    violation_count = 0

    if dtype == "binary":
        non_binary = values.filter(~values.is_in([0.0, 1.0]))
        violation_count = len(non_binary)
        if violation_count > 0:
            samples = non_binary.to_list()[:5]
            issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "dtype_violation",
                    "severity": "error",
                    "message": f"Binary indicator has values outside {{0, 1}}: {samples}",
                }
            )

    elif dtype == "count":
        negative = values.filter(values < 0)
        # Use the same tolerance as observation_families._nonneg_integer (atol=1e-6)
        # so values that pass stage 3 also pass at SSM build time.
        rounded = values.round(0)
        fractional = values.filter((values - rounded).abs() > 1e-6)
        violation_count = len(negative) + len(fractional)
        if len(negative) > 0:
            issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "dtype_violation",
                    "severity": "error",
                    "message": f"Count indicator has negative values: {negative.to_list()[:5]}",
                }
            )
        if len(fractional) > 0:
            issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "dtype_violation",
                    "severity": "error",
                    "message": (
                        f"Count indicator has fractional values: {fractional.to_list()[:5]}"
                    ),
                }
            )

    elif dtype == "continuous":
        n = len(values)
        if n >= MIN_OBSERVATIONS:
            q1_raw = values.quantile(0.25)
            q3_raw = values.quantile(0.75)
            assert isinstance(q1_raw, (int, float))
            assert isinstance(q3_raw, (int, float))
            q1 = float(q1_raw)
            q3 = float(q3_raw)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
                upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
                outliers = values.filter((values < lower) | (values > upper))
                violation_count = len(outliers)
                if violation_count > 0:
                    issues.append(
                        {
                            "indicator": ind_name,
                            "issue_type": "dtype_violation",
                            "severity": "warning",
                            "message": (
                                f"{violation_count} outlier(s) outside [{lower:.2f}, {upper:.2f}]"
                            ),
                        }
                    )

    return issues, violation_count


def _check_time_coverage(
    parsed_ts: pl.Series,
    model_clock_hours: float,
    ind_name: str,
) -> tuple[list[dict], float | None]:
    """Check if data spans enough time for temporal modeling.

    Returns:
        Tuple of (issues, time_coverage_ratio).
    """
    issues: list[dict] = []

    if len(parsed_ts) < 2:
        return issues, None

    ts_max = parsed_ts.max()
    ts_min = parsed_ts.min()
    assert isinstance(ts_max, datetime)
    assert isinstance(ts_min, datetime)
    time_span = ts_max - ts_min
    time_span_hours = time_span.total_seconds() / 3600
    min_hours = MIN_COVERAGE_PERIODS * model_clock_hours
    coverage_ratio = min(time_span_hours / min_hours, 1.0) if min_hours > 0 else None

    if time_span_hours < min_hours:
        issues.append(
            {
                "indicator": ind_name,
                "issue_type": "insufficient_coverage",
                "severity": "warning",
                "message": (
                    f"Time span {time_span_hours:.0f}h < required {min_hours}h "
                    f"({MIN_COVERAGE_PERIODS} x {model_clock_hours}h)"
                ),
            }
        )

    return issues, coverage_ratio


def _check_timestamp_gaps(
    parsed_ts: pl.Series,
    model_clock_hours: float,
    ind_name: str,
) -> tuple[list[dict], float | None]:
    """Check for excessively large gaps in timestamps.

    Returns:
        Tuple of (issues, max_gap_ratio).
    """
    issues: list[dict] = []

    if len(parsed_ts) < 3:
        return issues, None

    sorted_ts = parsed_ts.sort()
    diffs = sorted_ts.diff().drop_nulls()
    max_gap_raw = diffs.max()
    assert isinstance(max_gap_raw, timedelta)
    max_gap_hours = max_gap_raw.total_seconds() / 3600
    threshold = MAX_GAP_MULTIPLIER * model_clock_hours
    max_gap_ratio = max_gap_hours / threshold if threshold > 0 else None

    if max_gap_hours > threshold:
        issues.append(
            {
                "indicator": ind_name,
                "issue_type": "large_timestamp_gap",
                "severity": "warning",
                "message": (
                    f"Max consecutive gap {max_gap_hours:.0f}h > "
                    f"{MAX_GAP_MULTIPLIER}x {model_clock_hours}h ({threshold}h)"
                ),
            }
        )

    return issues, max_gap_ratio


def _check_hallucination_signals(
    values: pl.Series, dtype: str, ind_name: str
) -> tuple[list[dict], float, bool]:
    """Check for patterns suspicious of LLM hallucination.

    Returns:
        Tuple of (issues, duplicate_pct, arithmetic_sequence_detected).
    """
    issues: list[dict] = []
    n = len(values)
    duplicate_pct = 0.0
    arithmetic_sequence_detected = False

    if n < 2:
        return issues, duplicate_pct, arithmetic_sequence_detected

    vc = values.value_counts()
    max_count_raw = vc["count"].max()
    assert isinstance(max_count_raw, (int, float))
    max_count = int(max_count_raw)
    duplicate_pct = max_count / n if n > 0 else 0.0

    if dtype not in ("binary", "count"):
        var_raw = values.var()
        assert isinstance(var_raw, (int, float))
        variance = float(var_raw)
        if (
            variance is not None
            and variance > 0
            and max_count > n * HALLUCINATION_DUPLICATE_THRESHOLD
        ):
            most_common = vc.sort("count", descending=True).row(0)[0]
            issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "suspicious_pattern",
                    "severity": "warning",
                    "message": (
                        f">{HALLUCINATION_DUPLICATE_THRESHOLD * 100:.0f}% of values "
                        f"are {most_common} ({max_count}/{n})"
                    ),
                }
            )

    if n >= 5:
        sorted_vals = values.sort()
        diffs = sorted_vals.diff().drop_nulls()
        if diffs.n_unique() == 1:
            step = diffs[0]
            if step != 0:
                arithmetic_sequence_detected = True
                issues.append(
                    {
                        "indicator": ind_name,
                        "issue_type": "suspicious_pattern",
                        "severity": "warning",
                        "message": f"Values form arithmetic sequence with step {step}",
                    }
                )

    return issues, duplicate_pct, arithmetic_sequence_detected


def _check_construct_correlations(
    combined: pl.DataFrame,
    indicators: list[dict],
) -> list[dict]:
    """Check cross-indicator correlations within constructs."""
    issues: list[dict] = []

    construct_indicators: dict[str, list[str]] = {}
    for ind in indicators:
        cname = ind.get("construct_name", "")
        iname = ind.get("name", "")
        if cname and iname:
            construct_indicators.setdefault(cname, []).append(iname)

    for cname, ind_names in construct_indicators.items():
        if len(ind_names) < 2:
            continue

        for i in range(len(ind_names)):
            for j in range(i + 1, len(ind_names)):
                name_a, name_b = ind_names[i], ind_names[j]

                data_a = (
                    combined.filter(pl.col("indicator") == name_a)
                    .select(
                        _parsed_timestamp_expr(OBSERVATION_TIME_COLUMN).alias("ts"),
                        pl.col("value").cast(pl.Float64, strict=False).alias("value_a"),
                    )
                    .drop_nulls()
                )

                data_b = (
                    combined.filter(pl.col("indicator") == name_b)
                    .select(
                        _parsed_timestamp_expr(OBSERVATION_TIME_COLUMN).alias("ts"),
                        pl.col("value").cast(pl.Float64, strict=False).alias("value_b"),
                    )
                    .drop_nulls()
                )

                data_a = (
                    data_a.with_columns(pl.col("ts").dt.truncate("1d").alias("day"))
                    .group_by("day")
                    .agg(pl.col("value_a").mean())
                )
                data_b = (
                    data_b.with_columns(pl.col("ts").dt.truncate("1d").alias("day"))
                    .group_by("day")
                    .agg(pl.col("value_b").mean())
                )

                aligned = data_a.join(data_b, on="day", how="inner")

                if len(aligned) < MIN_ALIGNED_FOR_CFA:
                    continue

                r = aligned.select(pl.corr("value_a", "value_b")).item()

                if r is not None and not math.isnan(r) and r < 0:
                    issues.append(
                        {
                            "indicator": cname,
                            "issue_type": "low_construct_correlation",
                            "severity": "warning",
                            "message": (
                                f"Indicators {name_a} and {name_b} have negative "
                                f"daily correlation (r={r:.3f}), violating reflective "
                                f"measurement assumption"
                            ),
                        }
                    )

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# Rule check functions (return ValidationFindings)
# ══════════════════════════════════════════════════════════════════════════════


def _rule_missing(entry: IndicatorRuleInput) -> ValidationFindings:
    """Indicator declared in the causal spec but absent from extracted data."""
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
    """Indicator has rows but no numeric values after coercion."""
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
    """Timestamp parseability."""
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    issues = []
    for severity, message in _timestamp_issue_specs(ctx.n_total_ts, ctx.n_unparseable):
        issues.append(
            Issue(
                ctx.name,
                "unparseable_timestamps",
                severity,
                message,
                cell_key="n_unparseable_timestamps",
            )
        )
    return ValidationFindings(
        issues=issues,
        metrics={"n_unparseable_timestamps": ctx.n_unparseable},
    )


def _rule_sample_size(entry: IndicatorRuleInput) -> ValidationFindings:
    """Minimum observation count."""
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
    """Zero-variance check."""
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    issues = []
    if ctx.variance is not None and ctx.variance == 0:
        const_val = ctx.values.first()
        issues.append(
            Issue(
                ctx.name,
                "no_variance",
                "error",
                f"Zero variance (constant value = {const_val})",
                cell_key="variance",
            )
        )
    return ValidationFindings(issues=issues, metrics={"variance": ctx.variance})


def _rule_dtype_range(entry: IndicatorRuleInput) -> ValidationFindings:
    """Dtype range conformance."""
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    if not ctx.dtype:
        return ValidationFindings(metrics={"dtype_violations": 0})
    raw_issues, violation_count = _check_dtype_range(ctx.values, ctx.dtype, ctx.name)
    issues = _issues_from_raw(raw_issues, cell_key="dtype_violations")
    return ValidationFindings(issues=issues, metrics={"dtype_violations": violation_count})


def _rule_time_coverage(entry: IndicatorRuleInput) -> ValidationFindings:
    """Time span coverage."""
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    if ctx.is_time_invariant or ctx.model_clock_hours is None:
        return ValidationFindings(metrics={"time_coverage_ratio": None})
    raw_issues, ratio = _check_time_coverage(ctx.parsed_ts, ctx.model_clock_hours, ctx.name)
    issues = _issues_from_raw(raw_issues, cell_key="time_coverage_ratio")
    return ValidationFindings(issues=issues, metrics={"time_coverage_ratio": ratio})


def _rule_timestamp_gaps(entry: IndicatorRuleInput) -> ValidationFindings:
    """Max consecutive gap."""
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    if ctx.is_time_invariant or ctx.model_clock_hours is None:
        return ValidationFindings(metrics={"max_gap_ratio": None})
    raw_issues, ratio = _check_timestamp_gaps(ctx.parsed_ts, ctx.model_clock_hours, ctx.name)
    issues = _issues_from_raw(raw_issues, cell_key="max_gap_ratio")
    return ValidationFindings(issues=issues, metrics={"max_gap_ratio": ratio})


def _rule_hallucination_signals(entry: IndicatorRuleInput) -> ValidationFindings:
    """Suspicious LLM extraction patterns."""
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    raw_issues, duplicate_pct, arith_seq = _check_hallucination_signals(
        ctx.values, ctx.dtype or "continuous", ctx.name
    )
    issues = _issues_from_raw(
        raw_issues,
        cell_key=lambda raw_issue: (
            "arithmetic_sequence_detected"
            if "arithmetic sequence" in raw_issue["message"]
            else "duplicate_pct"
        ),
    )
    return ValidationFindings(
        issues=issues,
        metrics={"duplicate_pct": duplicate_pct, "arithmetic_sequence_detected": arith_seq},
    )


def _rule_construct_correlations(
    combined: pl.DataFrame, indicators: list[dict]
) -> ValidationFindings:
    """Cross-indicator construct correlations (dataset-wide)."""
    raw_issues = _check_construct_correlations(combined, indicators)
    issues = _issues_from_raw(raw_issues, cell_key="")
    return ValidationFindings(issues=issues)


# ══════════════════════════════════════════════════════════════════════════════
# Rule registry
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Reducer + runner
# ══════════════════════════════════════════════════════════════════════════════

# Metrics that get cell_status entries in per_indicator_health
_CELL_STATUS_KEYS = frozenset(
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
    """Reduce per-indicator findings into issues + keyed validation metrics.

    Cell statuses are derived purely from issues: for each metric key,
    the worst severity among matching issues wins (error > warning > ok).
    Rules own threshold logic; the reducer only aggregates.
    """
    all_issues: list[dict] = []
    indicator_health: dict[str, dict[str, Any]] = {}

    for ind_name in sorted(indicator_findings):
        findings_list = indicator_findings[ind_name]
        merged_metrics: dict[str, Any] = {}
        ind_issues: list[Issue] = []
        for f in findings_list:
            ind_issues.extend(f.issues)
            merged_metrics.update(f.metrics)

        for issue in ind_issues:
            all_issues.append(_issue_payload(issue))

        cell_statuses: dict[str, str] = {k: "ok" for k in _CELL_STATUS_KEYS if k in merged_metrics}
        for issue in ind_issues:
            if issue.cell_key in cell_statuses and cell_statuses[issue.cell_key] != "error":
                cell_statuses[issue.cell_key] = issue.severity

        indicator_health[ind_name] = {
            **{k: v for k, v in merged_metrics.items() if k in _CELL_STATUS_KEYS},
            "cell_statuses": cell_statuses,
        }

    return all_issues, indicator_health


def _build_indicator_context(
    ind_name: str,
    ind_data: pl.DataFrame,
    indicator_lookup: dict[str, dict],
    construct_lookup: dict[str, dict],
    model_clock_hours: float | None,
) -> IndicatorContext | None:
    """Build an IndicatorContext, or None if the indicator lacks numeric data."""
    values_df = ind_data.select(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()
    n_obs = len(values_df)
    if n_obs == 0:
        return None

    values = values_df["value"]
    variance: float | None = None
    try:
        _var = values.var()
        if isinstance(_var, timedelta):
            variance = _var.total_seconds()
        elif _var is not None:
            variance = float(_var)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        logger.info("Variance calculation failed for indicator %s", ind_name, exc_info=True)

    ind_meta = indicator_lookup.get(ind_name, {})
    dtype = ind_meta.get("measurement_dtype")
    construct_name = ind_meta.get("construct_name")
    construct_meta = construct_lookup.get(construct_name, {}) if construct_name else {}
    is_time_invariant = construct_meta.get("temporal_status") == "time_invariant"

    # Parse observation times once for all rules
    timestamps = ind_data[OBSERVATION_TIME_COLUMN]
    n_total_ts = len(timestamps)
    parsed = _parse_timestamp_series(timestamps)
    n_unparseable = parsed.null_count()
    parsed_ts = parsed.drop_nulls()

    return IndicatorContext(
        name=ind_name,
        ind_data=ind_data,
        values=values,
        n_obs=n_obs,
        variance=variance,
        dtype=dtype,
        is_time_invariant=is_time_invariant,
        model_clock_hours=model_clock_hours,
        parsed_ts=parsed_ts,
        n_unparseable=n_unparseable,
        n_total_ts=n_total_ts,
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
    ind_name: str,
    model_data: pl.DataFrame,
    indicator_lookup: dict[str, dict],
    health_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    """Compute the model-facing empirical profile for one indicator."""
    ind_model = model_data.filter(pl.col("indicator") == ind_name)
    values_df = ind_model.select(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()
    n_obs = len(values_df)
    if n_obs == 0:
        return None

    values = values_df["value"]
    mean = _float_or_none(values.mean())
    std = _float_or_none(values.std())
    variance = _float_or_none(values.var())
    min_value = _float_or_none(values.min())
    max_value = _float_or_none(values.max())
    q25 = _float_or_none(values.quantile(0.25))
    q50 = _float_or_none(values.quantile(0.50))
    q75 = _float_or_none(values.quantile(0.75))

    numeric_values = [float(v) for v in values.to_list()]
    zero_fraction = (
        float(sum(1 for v in numeric_values if math.isclose(v, 0.0, abs_tol=1e-12)) / n_obs)
        if n_obs > 0
        else None
    )
    looks_integer_valued = all(math.isclose(v, round(v), abs_tol=1e-8) for v in numeric_values)
    is_nonnegative = min_value >= 0 if min_value is not None else None
    is_unit_interval = (
        min_value >= 0 and max_value <= 1
        if min_value is not None and max_value is not None
        else None
    )
    variance_to_mean_ratio = (
        variance / mean if variance is not None and mean is not None and mean > 0 else None
    )

    return {
        "measurement_dtype": indicator_lookup.get(ind_name, {}).get("measurement_dtype"),
        "n_obs": n_obs,
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "variance": variance,
        "time_coverage_ratio": _float_or_none(health_metrics.get("time_coverage_ratio")),
        "max_gap_ratio": _float_or_none(health_metrics.get("max_gap_ratio")),
        "dtype_violations": health_metrics.get("dtype_violations"),
        "duplicate_pct": _float_or_none(health_metrics.get("duplicate_pct")),
        "arithmetic_sequence_detected": bool(
            health_metrics.get("arithmetic_sequence_detected", False)
        ),
        "n_unparseable_timestamps": health_metrics.get("n_unparseable_timestamps"),
        "zero_fraction": zero_fraction,
        "is_nonnegative": is_nonnegative,
        "is_unit_interval": is_unit_interval,
        "looks_integer_valued": looks_integer_valued,
        "variance_to_mean_ratio": variance_to_mean_ratio,
    }


def build_indicator_audits(
    *,
    indicator_names: set[str],
    indicator_lookup: dict[str, dict],
    model_data: pl.DataFrame,
    indicator_issues: list[dict[str, Any]],
    indicator_health: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the keyed Stage 3 indicator audit map."""
    issues_by_indicator: dict[str, list[dict[str, Any]]] = {name: [] for name in indicator_names}
    for issue in indicator_issues:
        issue_indicator = issue.get("indicator")
        if issue_indicator in issues_by_indicator:
            issues_by_indicator[issue_indicator].append(issue)

    audits: dict[str, dict[str, Any]] = {}
    for ind_name in sorted(indicator_names):
        health_metrics = indicator_health.get(ind_name, {})
        audits[ind_name] = {
            "profile": _compute_empirical_profile(
                ind_name,
                model_data,
                indicator_lookup,
                health_metrics,
            ),
            "validation": {
                "issues": issues_by_indicator.get(ind_name, []),
                "checks": dict(health_metrics.get("cell_statuses", {})),
            },
        }
    return audits


def run_rules(
    rules: list[ValidationRule],
    ctx: ValidationContext,
) -> tuple[list[dict], dict[str, dict[str, Any]], list[dict]]:
    """Run all validation rules and reduce findings.

    Returns:
        (indicator_issues, indicator_health, dataset_issues)
    """
    indicator_rules = [r for r in rules if r.scope == "indicator"]
    dataset_rules = [r for r in rules if r.scope == "dataset"]

    indicator_findings: dict[str, list[ValidationFindings]] = {}

    for ind_name, ind_data, indicator_ctx in ctx.iter_indicators():
        rule_input = IndicatorRuleInput(ind_name, ind_data, indicator_ctx)
        indicator_findings[ind_name] = [rule.check(rule_input) for rule in indicator_rules]

    dataset_issues: list[dict] = []
    for rule in dataset_rules:
        f = rule.check(ctx.combined, ctx.indicators)
        for issue in f.issues:
            dataset_issues.append(_issue_payload(issue))

    indicator_issues, indicator_health = reduce_findings(indicator_findings)
    return indicator_issues, indicator_health, dataset_issues


# ══════════════════════════════════════════════════════════════════════════════
# Task
# ══════════════════════════════════════════════════════════════════════════════


@task(cache_policy=INPUTS, result_serializer="json")
def validate_extraction(
    causal_spec: dict,
    dataframes: list[pl.DataFrame],
) -> dict:
    """Validate semantic properties of extracted data.

    Runs all ``RULES`` against the extracted data and reduces findings
    into a keyed indicator audit map plus dataset-level issues.

    Args:
        causal_spec: The full causal spec with measurement model
        dataframes: List of DataFrames with columns (indicator, value, anchor_time)

    Returns:
        Dict with:
            - is_valid: bool
            - indicators: per-indicator profile + validation
            - dataset_issues: cross-indicator validation findings
    """
    dataframes = [df for df in dataframes if df is not None and not df.is_empty()]
    if not dataframes:
        return _no_data_validation_result()

    combined = pl.concat(dataframes, how="vertical")

    if combined.is_empty():
        return _no_data_validation_result()

    from causal_ssm_agent.utils.causal_spec import get_constructs, get_indicators

    indicators = get_indicators(causal_spec)
    indicator_names: set[str] = {ind["name"] for ind in indicators if ind.get("name")}
    indicator_lookup = {ind["name"]: ind for ind in indicators if ind.get("name")}

    constructs = get_constructs(causal_spec)
    construct_lookup = {c["name"]: c for c in constructs if c.get("name")}

    model_clock_str = causal_spec.get("measurement", {}).get("model_clock")
    model_clock_hours: float | None = None
    if model_clock_str:
        import contextlib

        from causal_ssm_agent.orchestrator.schemas import parse_duration_to_hours

        with contextlib.suppress(ValueError):
            model_clock_hours = parse_duration_to_hours(model_clock_str)

    validation_ctx = ValidationContext(
        combined=combined,
        indicators=indicators,
        indicator_names=indicator_names,
        indicator_lookup=indicator_lookup,
        construct_lookup=construct_lookup,
        model_clock_hours=model_clock_hours,
    )

    indicator_issues, indicator_health, dataset_issues = run_rules(
        RULES,
        validation_ctx,
    )

    indicator_audits = build_indicator_audits(
        indicator_names=indicator_names,
        indicator_lookup=indicator_lookup,
        model_data=combined,
        indicator_issues=indicator_issues,
        indicator_health=indicator_health,
    )

    all_issues = [*indicator_issues, *dataset_issues]
    status = derive_validation_status(all_issues)

    return {
        "is_valid": status["is_valid"],
        "indicators": indicator_audits,
        "dataset_issues": dataset_issues,
    }


def _parsed_timestamp_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8, strict=False)
    candidates: list[pl.Expr] = []
    for fmt, has_tz in _TIMESTAMP_FORMATS:
        parsed = text.str.to_datetime(format=fmt, strict=False)
        if has_tz:
            parsed = parsed.dt.replace_time_zone(None)
        candidates.append(parsed)
    return pl.coalesce(candidates)


def _parse_timestamp_series(timestamps: pl.Series) -> pl.Series:
    return pl.DataFrame({"timestamp": timestamps}).select(_parsed_timestamp_expr("timestamp"))[
        "timestamp"
    ]
