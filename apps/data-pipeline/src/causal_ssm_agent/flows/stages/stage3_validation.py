"""Stage 3: Validate extracted data.

Validation checks (semantic only - Polars handles structural validation):
1. Variance: Indicator has variance > 0 (constant values = zero information)
2. Sample size: Enough observations for temporal modeling
3. Timestamps: Parseable and covering sufficient time span
4. Dtype range: Values match declared measurement dtype
5. Timestamp gaps: No excessively large gaps between observations
6. Hallucination signals: Suspicious patterns from LLM extraction
7. Construct correlations: Cross-indicator coherence within constructs

See docs/reference/pipeline.md for full specification.
"""

import math
from datetime import datetime, timedelta

import polars as pl
from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.flows import get_prefect_logger

logger = get_prefect_logger(__name__)

# Minimum observations for temporal modeling
MIN_OBSERVATIONS = 10  # Reasonable minimum for temporal modeling

# Validation constants
MIN_COVERAGE_PERIODS = 10
MAX_GAP_MULTIPLIER = 5
OUTLIER_IQR_MULTIPLIER = 3.0
MIN_ALIGNED_FOR_CFA = 10
HALLUCINATION_DUPLICATE_THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
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

    try:
        parsed = timestamps.str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S%z", strict=False
        ).dt.replace_time_zone(None)
    except pl.exceptions.ComputeError:
        # Polars can't infer any format — treat all as unparseable
        parsed = pl.Series("timestamp", [None] * n_total, dtype=pl.Datetime("us"))
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

    # Compute duplicate percentage for all types
    vc = values.value_counts()
    max_count_raw = vc["count"].max()
    assert isinstance(max_count_raw, (int, float))
    max_count = int(max_count_raw)
    duplicate_pct = max_count / n if n > 0 else 0.0

    # Excessive duplicates (only for continuous/ordinal data with variance > 0)
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

    # Arithmetic sequence check
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

    # Group indicators by construct
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
                        pl.col("timestamp")
                        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z", strict=False)
                        .dt.replace_time_zone(None)
                        .alias("ts"),
                        pl.col("value").cast(pl.Float64, strict=False).alias("value_a"),
                    )
                    .drop_nulls()
                )

                data_b = (
                    combined.filter(pl.col("indicator") == name_b)
                    .select(
                        pl.col("timestamp")
                        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z", strict=False)
                        .dt.replace_time_zone(None)
                        .alias("ts"),
                        pl.col("value").cast(pl.Float64, strict=False).alias("value_b"),
                    )
                    .drop_nulls()
                )

                # Bucket to daily resolution so indicators with different
                # sub-day timestamps can still align for correlation checks.
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
# TASKS
# ══════════════════════════════════════════════════════════════════════════════


@task(cache_policy=INPUTS, result_serializer="json")
def validate_extraction(
    causal_spec: dict,
    dataframes: list[pl.DataFrame],
) -> dict:
    """Validate semantic properties of extracted data.

    Checks extracted data for:
    - Variance > 0 (constant values are uninformative)
    - Sample size (enough observations for temporal modeling)
    - Timestamp parseability and coverage
    - Dtype range conformance
    - Timestamp gap detection
    - Hallucination signal detection
    - Cross-indicator construct correlations

    Args:
        causal_spec: The full causal spec with measurement model
        dataframes: List of DataFrames with columns (indicator, value, timestamp)

    Returns:
        Dict with:
            - is_valid: bool
            - issues: list of {indicator, issue_type, severity, message}
            - per_indicator_health: list of per-indicator metrics
    """
    # Filter out empty/None
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

    issues: list[dict] = []
    per_indicator_health: list[dict] = []

    # Check each indicator in the extracted data
    for ind_name in indicator_names:
        ind_data = combined.filter(pl.col("indicator") == ind_name)

        if ind_data.is_empty():
            issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "missing",
                    "severity": "warning",
                    "message": "No data extracted for this indicator",
                }
            )
            continue

        # Coerce values to numeric for validation
        values = ind_data.select(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()

        n_obs = len(values)

        if n_obs == 0:
            issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "no_numeric",
                    "severity": "error",
                    "message": "No numeric values extracted",
                }
            )
            continue

        # Check sample size
        if n_obs < MIN_OBSERVATIONS:
            issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "low_n",
                    "severity": "warning",
                    "message": f"Only {n_obs} observations (recommend >= {MIN_OBSERVATIONS})",
                }
            )

        # Check variance
        variance: float | None = None
        try:
            _var = values["value"].var()
            variance = float(_var) if _var is not None else None
            if variance is not None and variance == 0:
                const_val = values["value"].first()
                issues.append(
                    {
                        "indicator": ind_name,
                        "issue_type": "no_variance",
                        "severity": "error",
                        "message": f"Zero variance (constant value = {const_val})",
                    }
                )
        except Exception as e:
            logger.debug("Variance check failed for indicator: %s", e)

        # ── Validation checks with metric collection ─────────────────────

        # Get indicator metadata for dtype/construct lookups
        ind_meta = indicator_lookup.get(ind_name, {})
        dtype = ind_meta.get("measurement_dtype")
        construct_name = ind_meta.get("construct_name")
        construct_meta = construct_lookup.get(construct_name, {}) if construct_name else {}
        is_time_invariant = construct_meta.get("temporal_status") == "time_invariant"

        # Get model_clock_hours from measurement model
        model_clock_str = causal_spec.get("measurement", {}).get("model_clock")
        model_clock_hours: float | None = None
        if model_clock_str:
            import contextlib

            from causal_ssm_agent.orchestrator.schemas import parse_duration_to_hours

            with contextlib.suppress(ValueError):
                model_clock_hours = parse_duration_to_hours(model_clock_str)

        # 1. Timestamp parseability
        ts_issues, parsed_ts = _check_timestamps(ind_data, ind_name)
        issues.extend(ts_issues)

        # 2. Dtype range conformance
        dtype_violations = 0
        if dtype:
            dtype_issues, dtype_violations = _check_dtype_range(values["value"], dtype, ind_name)
            issues.extend(dtype_issues)

        # 3-4. Time coverage and gaps (skip for time-invariant constructs)
        time_coverage_ratio: float | None = None
        max_gap_ratio: float | None = None
        if not is_time_invariant and model_clock_hours is not None:
            coverage_issues, time_coverage_ratio = _check_time_coverage(
                parsed_ts, model_clock_hours, ind_name
            )
            issues.extend(coverage_issues)
            gap_issues, max_gap_ratio = _check_timestamp_gaps(
                parsed_ts, model_clock_hours, ind_name
            )
            issues.extend(gap_issues)

        # 5. Hallucination signals
        halluc_issues, duplicate_pct, arithmetic_sequence_detected = _check_hallucination_signals(
            values["value"], dtype or "continuous", ind_name
        )
        issues.extend(halluc_issues)

        # Derive per-cell pass/fail statuses for frontend color coding.
        # These mirror the issue severity decisions above so the frontend
        # never needs to re-derive domain thresholds.
        cell_statuses: dict[str, str] = {
            "n_obs": "warning" if n_obs < MIN_OBSERVATIONS else "ok",
            "variance": ("error" if variance is not None and variance == 0 else "ok"),
            "time_coverage_ratio": (
                "warning" if time_coverage_ratio is not None and time_coverage_ratio < 1.0 else "ok"
            ),
            "max_gap_ratio": (
                "warning" if max_gap_ratio is not None and max_gap_ratio > 1.0 else "ok"
            ),
            "dtype_violations": (
                "error"
                if dtype_violations > 0 and dtype in ("binary", "count")
                else "warning"
                if dtype_violations > 0
                else "ok"
            ),
            "duplicate_pct": (
                "warning"
                if duplicate_pct > HALLUCINATION_DUPLICATE_THRESHOLD
                and dtype not in ("binary", "count")
                and variance is not None
                and variance > 0
                else "ok"
            ),
            "arithmetic_sequence_detected": ("warning" if arithmetic_sequence_detected else "ok"),
        }

        per_indicator_health.append(
            {
                "indicator": ind_name,
                "n_obs": n_obs,
                "variance": variance,
                "time_coverage_ratio": time_coverage_ratio,
                "max_gap_ratio": max_gap_ratio,
                "dtype_violations": dtype_violations,
                "duplicate_pct": duplicate_pct,
                "arithmetic_sequence_detected": arithmetic_sequence_detected,
                "cell_statuses": cell_statuses,
            }
        )

    # 6. Cross-indicator construct correlations
    issues.extend(_check_construct_correlations(combined, indicators))

    errors = [i for i in issues if i["severity"] == "error"]
    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "issues": issues,
        "per_indicator_health": per_indicator_health,
    }
