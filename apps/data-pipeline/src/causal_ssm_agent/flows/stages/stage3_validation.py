"""Stage 3: Validate extracted data.

Validation is expressed as composable ``ValidationRule`` entries. Each rule
returns ``ValidationFindings`` (issues with ``cell_key`` + raw metrics).
A single ``reduce_findings()`` derives cell statuses from issues, so
threshold logic lives inside rules and status derivation is centralized.

Adding a new validation check = appending one entry to ``RULES``.

See docs/reference/pipeline.md for full specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import polars as pl
from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.flows import get_prefect_logger

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

_TIMESTAMP_FORMATS: tuple[tuple[str | None, bool], ...] = (
    ("%Y-%m-%d %H:%M:%S", False),
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
class ValidationContext:
    """Dataset-level validation context with helpers for per-indicator views."""

    combined: pl.DataFrame
    indicators: list[dict]
    indicator_names: set[str]
    indicator_lookup: dict[str, dict]
    construct_lookup: dict[str, dict]
    model_clock_hours: float | None

    def iter_indicators(self):
        for ind_name in self.indicator_names:
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
    check: Any  # (IndicatorContext) -> ValidationFindings or dataset check callable


# ══════════════════════════════════════════════════════════════════════════════
# Low-level check functions (preserved for direct testability)
# ══════════════════════════════════════════════════════════════════════════════


def _check_timestamps(ind_data: pl.DataFrame, ind_name: str) -> tuple[list[dict], pl.Series]:
    """Check timestamp parseability.

    Returns:
        Tuple of (issues, parsed_timestamps_without_nulls).
    """
    issues: list[dict] = []
    timestamps = ind_data["timestamp"]
    n_total = len(timestamps)

    if n_total == 0:
        return issues, pl.Series("timestamp", [], dtype=pl.Datetime("us"))

    parsed = _parse_timestamp_series(timestamps)
    n_unparseable = parsed.null_count()

    if n_unparseable == n_total:
        issues.append(
            {
                "indicator": ind_name,
                "issue_type": "unparseable_timestamps",
                "severity": "error",
                "message": f"All {n_total} timestamps are unparseable",
            }
        )
    elif n_unparseable > n_total * 0.5:
        issues.append(
            {
                "indicator": ind_name,
                "issue_type": "unparseable_timestamps",
                "severity": "warning",
                "message": f"{n_unparseable}/{n_total} timestamps are unparseable (>50%)",
            }
        )

    return issues, parsed.drop_nulls()


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
        fractional = values.filter((values % 1) != 0)
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
            assert isinstance(q1_raw, (int, float)) and isinstance(q3_raw, (int, float))
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
    assert isinstance(ts_max, datetime) and isinstance(ts_min, datetime)
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
                        _parsed_timestamp_expr("timestamp").alias("ts"),
                        pl.col("value").cast(pl.Float64, strict=False).alias("value_a"),
                    )
                    .drop_nulls()
                )

                data_b = (
                    combined.filter(pl.col("indicator") == name_b)
                    .select(
                        _parsed_timestamp_expr("timestamp").alias("ts"),
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


def _rule_timestamps(ctx: IndicatorContext) -> ValidationFindings:
    """Timestamp parseability."""
    issues = []
    if ctx.n_total_ts > 0:
        if ctx.n_unparseable == ctx.n_total_ts:
            issues.append(
                Issue(
                    ctx.name,
                    "unparseable_timestamps",
                    "error",
                    f"All {ctx.n_total_ts} timestamps are unparseable",
                    cell_key="n_obs",
                )
            )
        elif ctx.n_unparseable > ctx.n_total_ts * 0.5:
            issues.append(
                Issue(
                    ctx.name,
                    "unparseable_timestamps",
                    "warning",
                    f"{ctx.n_unparseable}/{ctx.n_total_ts} timestamps are unparseable (>50%)",
                    cell_key="n_obs",
                )
            )
    return ValidationFindings(issues=issues)


def _rule_sample_size(ctx: IndicatorContext) -> ValidationFindings:
    """Minimum observation count."""
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


def _rule_variance(ctx: IndicatorContext) -> ValidationFindings:
    """Zero-variance check."""
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


def _rule_dtype_range(ctx: IndicatorContext) -> ValidationFindings:
    """Dtype range conformance."""
    if not ctx.dtype:
        return ValidationFindings(metrics={"dtype_violations": 0})
    raw_issues, violation_count = _check_dtype_range(ctx.values, ctx.dtype, ctx.name)
    issues = [
        Issue(
            d["indicator"],
            d["issue_type"],
            d["severity"],
            d["message"],
            cell_key="dtype_violations",
        )
        for d in raw_issues
    ]
    return ValidationFindings(issues=issues, metrics={"dtype_violations": violation_count})


def _rule_time_coverage(ctx: IndicatorContext) -> ValidationFindings:
    """Time span coverage."""
    if ctx.is_time_invariant or ctx.model_clock_hours is None:
        return ValidationFindings(metrics={"time_coverage_ratio": None})
    raw_issues, ratio = _check_time_coverage(ctx.parsed_ts, ctx.model_clock_hours, ctx.name)
    issues = [
        Issue(
            d["indicator"],
            d["issue_type"],
            d["severity"],
            d["message"],
            cell_key="time_coverage_ratio",
        )
        for d in raw_issues
    ]
    return ValidationFindings(issues=issues, metrics={"time_coverage_ratio": ratio})


def _rule_timestamp_gaps(ctx: IndicatorContext) -> ValidationFindings:
    """Max consecutive gap."""
    if ctx.is_time_invariant or ctx.model_clock_hours is None:
        return ValidationFindings(metrics={"max_gap_ratio": None})
    raw_issues, ratio = _check_timestamp_gaps(ctx.parsed_ts, ctx.model_clock_hours, ctx.name)
    issues = [
        Issue(
            d["indicator"], d["issue_type"], d["severity"], d["message"], cell_key="max_gap_ratio"
        )
        for d in raw_issues
    ]
    return ValidationFindings(issues=issues, metrics={"max_gap_ratio": ratio})


def _rule_hallucination_signals(ctx: IndicatorContext) -> ValidationFindings:
    """Suspicious LLM extraction patterns."""
    raw_issues, duplicate_pct, arith_seq = _check_hallucination_signals(
        ctx.values, ctx.dtype or "continuous", ctx.name
    )
    # Split issues by cell_key: duplicate → "duplicate_pct", sequence → "arithmetic_sequence_detected"
    issues = []
    for d in raw_issues:
        if "arithmetic sequence" in d["message"]:
            cell_key = "arithmetic_sequence_detected"
        else:
            cell_key = "duplicate_pct"
        issues.append(
            Issue(d["indicator"], d["issue_type"], d["severity"], d["message"], cell_key=cell_key)
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
    issues = [
        Issue(d["indicator"], d["issue_type"], d["severity"], d["message"], cell_key="")
        for d in raw_issues
    ]
    return ValidationFindings(issues=issues)


# ══════════════════════════════════════════════════════════════════════════════
# Rule registry
# ══════════════════════════════════════════════════════════════════════════════

RULES: list[ValidationRule] = [
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
        "time_coverage_ratio",
        "max_gap_ratio",
        "dtype_violations",
        "duplicate_pct",
        "arithmetic_sequence_detected",
    }
)


def reduce_findings(
    indicator_findings: dict[str, list[ValidationFindings]],
) -> tuple[list[dict], list[dict]]:
    """Reduce per-indicator findings into issues list + per_indicator_health.

    Cell statuses are derived purely from issues: for each metric key,
    the worst severity among matching issues wins (error > warning > ok).
    Rules own threshold logic; the reducer only aggregates.
    """
    all_issues: list[dict] = []
    per_indicator_health: list[dict] = []

    for ind_name, findings_list in indicator_findings.items():
        merged_metrics: dict[str, Any] = {}
        ind_issues: list[Issue] = []
        for f in findings_list:
            ind_issues.extend(f.issues)
            merged_metrics.update(f.metrics)

        # Convert Issue dataclasses to dicts for output
        for issue in ind_issues:
            all_issues.append(
                {
                    "indicator": issue.indicator,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "message": issue.message,
                }
            )

        # Derive cell statuses from issues (error > warning > ok)
        cell_statuses: dict[str, str] = {k: "ok" for k in _CELL_STATUS_KEYS if k in merged_metrics}
        for issue in ind_issues:
            if issue.cell_key in cell_statuses and cell_statuses[issue.cell_key] != "error":
                cell_statuses[issue.cell_key] = issue.severity

        per_indicator_health.append(
            {
                "indicator": ind_name,
                **{k: v for k, v in merged_metrics.items() if k in _CELL_STATUS_KEYS},
                "cell_statuses": cell_statuses,
            }
        )

    return all_issues, per_indicator_health


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
        variance = float(_var) if _var is not None else None
    except Exception:
        pass

    ind_meta = indicator_lookup.get(ind_name, {})
    dtype = ind_meta.get("measurement_dtype")
    construct_name = ind_meta.get("construct_name")
    construct_meta = construct_lookup.get(construct_name, {}) if construct_name else {}
    is_time_invariant = construct_meta.get("temporal_status") == "time_invariant"

    # Parse timestamps once for all rules
    timestamps = ind_data["timestamp"]
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


def run_rules(
    rules: list[ValidationRule],
    ctx: ValidationContext,
) -> tuple[list[dict], list[dict]]:
    """Run all validation rules and reduce findings.

    Returns:
        (all_issues, per_indicator_health) — same shape as before.
    """
    indicator_rules = [r for r in rules if r.scope == "indicator"]
    dataset_rules = [r for r in rules if r.scope == "dataset"]

    # Collect early-exit issues (missing, no_numeric) separately
    early_issues: list[dict] = []
    indicator_findings: dict[str, list[ValidationFindings]] = {}

    for ind_name, ind_data, indicator_ctx in ctx.iter_indicators():
        if ind_data.is_empty():
            early_issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "missing",
                    "severity": "warning",
                    "message": "No data extracted for this indicator",
                }
            )
            continue

        if indicator_ctx is None:
            early_issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "no_numeric",
                    "severity": "error",
                    "message": "No numeric values extracted",
                }
            )
            continue

        findings: list[ValidationFindings] = []
        for rule in indicator_rules:
            findings.append(rule.check(indicator_ctx))
        indicator_findings[ind_name] = findings

    # Run dataset-wide rules
    dataset_issues: list[dict] = []
    for rule in dataset_rules:
        f = rule.check(ctx.combined, ctx.indicators)
        for issue in f.issues:
            dataset_issues.append(
                {
                    "indicator": issue.indicator,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "message": issue.message,
                }
            )

    all_issues, per_indicator_health = reduce_findings(indicator_findings)
    all_issues = early_issues + all_issues + dataset_issues

    return all_issues, per_indicator_health


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
    into issues + per_indicator_health with centralized cell-status derivation.

    Args:
        causal_spec: The full causal spec with measurement model
        dataframes: List of DataFrames with columns (indicator, value, timestamp)

    Returns:
        Dict with:
            - is_valid: bool
            - issues: list of {indicator, issue_type, severity, message}
            - per_indicator_health: list of per-indicator metrics
    """
    dataframes = [df for df in dataframes if df is not None and not df.is_empty()]
    if not dataframes:
        return {
            "is_valid": False,
            "issues": [
                {
                    "indicator": "all",
                    "issue_type": "no_data",
                    "severity": "error",
                    "message": "No data extracted",
                }
            ],
            "per_indicator_health": [],
        }

    combined = pl.concat(dataframes, how="vertical")

    if combined.is_empty():
        return {
            "is_valid": False,
            "issues": [
                {
                    "indicator": "all",
                    "issue_type": "no_data",
                    "severity": "error",
                    "message": "No data extracted",
                }
            ],
            "per_indicator_health": [],
        }

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

    issues, per_indicator_health = run_rules(
        RULES,
        validation_ctx,
    )

    errors = [i for i in issues if i["severity"] == "error"]
    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "issues": issues,
        "per_indicator_health": per_indicator_health,
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
