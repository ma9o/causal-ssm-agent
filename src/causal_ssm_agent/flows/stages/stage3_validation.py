"""Stage 3: Validate and aggregate extracted data.

Validation checks (semantic only - Polars handles structural validation):
1. Variance: Indicator has variance > 0 (constant values = zero information)
2. Sample size: Enough observations for temporal modeling
3. Timestamps: Parseable and covering sufficient time span
4. Dtype range: Values match declared measurement dtype
5. Timestamp gaps: No excessively large gaps between observations
6. Hallucination signals: Suspicious patterns from LLM extraction
7. Construct correlations: Cross-indicator coherence within constructs

Aggregation: Workers extract at the finest resolution visible in their chunk.
aggregate_measurements() buckets timestamps and applies each indicator's
aggregation function to reach measurement_granularity. SSM discretization
then handles measurement_granularity -> continuous time.

See docs/reference/pipeline.md for full specification.
"""

import logging
import math
from typing import TYPE_CHECKING

import polars as pl
from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.orchestrator.schemas import GRANULARITY_HOURS
from causal_ssm_agent.utils.aggregations import aggregate_worker_measurements

if TYPE_CHECKING:
    from causal_ssm_agent.workers.agents import WorkerResult

logger = logging.getLogger(__name__)

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


def _check_timestamps(
    ind_data: pl.DataFrame, ind_name: str
) -> tuple[list[dict], pl.Series]:
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
        parsed = timestamps.str.to_datetime(strict=False)
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


def _check_dtype_range(
    values: pl.Series, dtype: str, ind_name: str
) -> list[dict]:
    """Check values conform to declared measurement dtype."""
    issues: list[dict] = []

    if dtype == "binary":
        non_binary = values.filter(~values.is_in([0.0, 1.0]))
        if len(non_binary) > 0:
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
        if len(negative) > 0:
            issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "dtype_violation",
                    "severity": "error",
                    "message": f"Count indicator has negative values: {negative.to_list()[:5]}",
                }
            )
        fractional = values.filter((values % 1) != 0)
        if len(fractional) > 0:
            issues.append(
                {
                    "indicator": ind_name,
                    "issue_type": "dtype_violation",
                    "severity": "error",
                    "message": (
                        f"Count indicator has fractional values: "
                        f"{fractional.to_list()[:5]}"
                    ),
                }
            )

    elif dtype == "continuous":
        n = len(values)
        if n >= MIN_OBSERVATIONS:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
                upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
                outliers = values.filter((values < lower) | (values > upper))
                if len(outliers) > 0:
                    issues.append(
                        {
                            "indicator": ind_name,
                            "issue_type": "dtype_violation",
                            "severity": "warning",
                            "message": (
                                f"{len(outliers)} outlier(s) outside "
                                f"[{lower:.2f}, {upper:.2f}]"
                            ),
                        }
                    )

    return issues


def _check_time_coverage(
    parsed_ts: pl.Series,
    causal_granularity: str | None,
    ind_name: str,
) -> list[dict]:
    """Check if data spans enough time for temporal modeling."""
    issues: list[dict] = []

    if causal_granularity is None:
        return issues

    gran_hours = GRANULARITY_HOURS.get(causal_granularity)
    if gran_hours is None:
        return issues

    if len(parsed_ts) < 2:
        return issues

    time_span = parsed_ts.max() - parsed_ts.min()
    time_span_hours = time_span.total_seconds() / 3600
    min_hours = MIN_COVERAGE_PERIODS * gran_hours

    if time_span_hours < min_hours:
        issues.append(
            {
                "indicator": ind_name,
                "issue_type": "insufficient_coverage",
                "severity": "warning",
                "message": (
                    f"Time span {time_span_hours:.0f}h < required {min_hours}h "
                    f"({MIN_COVERAGE_PERIODS} x {causal_granularity})"
                ),
            }
        )

    return issues


def _check_timestamp_gaps(
    parsed_ts: pl.Series,
    causal_granularity: str | None,
    ind_name: str,
) -> list[dict]:
    """Check for excessively large gaps in timestamps."""
    issues: list[dict] = []

    if causal_granularity is None:
        return issues

    gran_hours = GRANULARITY_HOURS.get(causal_granularity)
    if gran_hours is None:
        return issues

    if len(parsed_ts) < 3:
        return issues

    sorted_ts = parsed_ts.sort()
    diffs = sorted_ts.diff().drop_nulls()
    max_gap = diffs.max()
    max_gap_hours = max_gap.total_seconds() / 3600
    threshold = MAX_GAP_MULTIPLIER * gran_hours

    if max_gap_hours > threshold:
        issues.append(
            {
                "indicator": ind_name,
                "issue_type": "large_timestamp_gap",
                "severity": "warning",
                "message": (
                    f"Max consecutive gap {max_gap_hours:.0f}h > "
                    f"{MAX_GAP_MULTIPLIER}x {causal_granularity} ({threshold}h)"
                ),
            }
        )

    return issues


def _check_hallucination_signals(
    values: pl.Series, dtype: str, ind_name: str
) -> list[dict]:
    """Check for patterns suspicious of LLM hallucination."""
    issues: list[dict] = []
    n = len(values)
    if n < 2:
        return issues

    # Excessive duplicates (only for continuous/ordinal data with variance > 0)
    if dtype not in ("binary", "count"):
        variance = values.var()
        if variance is not None and variance > 0:
            vc = values.value_counts()
            max_count = vc["count"].max()
            if max_count > n * HALLUCINATION_DUPLICATE_THRESHOLD:
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
                issues.append(
                    {
                        "indicator": ind_name,
                        "issue_type": "suspicious_pattern",
                        "severity": "warning",
                        "message": f"Values form arithmetic sequence with step {step}",
                    }
                )

    return issues


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
                        pl.col("timestamp").str.to_datetime(strict=False).alias("ts"),
                        pl.col("value").cast(pl.Float64, strict=False).alias("value_a"),
                    )
                    .drop_nulls()
                )

                data_b = (
                    combined.filter(pl.col("indicator") == name_b)
                    .select(
                        pl.col("timestamp").str.to_datetime(strict=False).alias("ts"),
                        pl.col("value").cast(pl.Float64, strict=False).alias("value_b"),
                    )
                    .drop_nulls()
                )

                aligned = data_a.join(data_b, on="ts", how="inner")

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
                                f"correlation (r={r:.3f}), violating reflective "
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
    worker_results: list["WorkerResult"],
) -> dict:
    """Validate semantic properties of extracted data.

    Checks raw worker extractions for:
    - Variance > 0 (constant values are uninformative)
    - Sample size (enough observations for temporal modeling)
    - Timestamp parseability and coverage
    - Dtype range conformance
    - Timestamp gap detection
    - Hallucination signal detection
    - Cross-indicator construct correlations

    Args:
        causal_spec: The full causal spec with measurement model
        worker_results: List of WorkerResults from Stage 2

    Returns:
        Dict with:
            - is_valid: bool
            - issues: list of {indicator, issue_type, severity, message}
    """
    # Concatenate all worker dataframes
    dataframes = [wr.dataframe for wr in worker_results if wr.dataframe is not None]
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
        }

    indicators = causal_spec.get("measurement", {}).get("indicators", [])
    indicator_names = {ind.get("name") for ind in indicators if ind.get("name")}
    indicator_lookup = {ind["name"]: ind for ind in indicators if ind.get("name")}

    constructs = causal_spec.get("latent", {}).get("constructs", [])
    construct_lookup = {c["name"]: c for c in constructs if c.get("name")}

    issues: list[dict] = []

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
        try:
            variance = values["value"].var()
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

        # ── New validation checks ──────────────────────────────────────────

        # Get indicator metadata for dtype/construct lookups
        ind_meta = indicator_lookup.get(ind_name, {})
        dtype = ind_meta.get("measurement_dtype")
        construct_name = ind_meta.get("construct_name")
        construct_meta = (
            construct_lookup.get(construct_name, {}) if construct_name else {}
        )
        causal_gran = construct_meta.get("causal_granularity")
        is_time_invariant = construct_meta.get("temporal_status") == "time_invariant"

        # 1. Timestamp parseability
        ts_issues, parsed_ts = _check_timestamps(ind_data, ind_name)
        issues.extend(ts_issues)

        # 2. Dtype range conformance
        if dtype:
            issues.extend(_check_dtype_range(values["value"], dtype, ind_name))

        # 3-4. Time coverage and gaps (skip for time-invariant constructs)
        if not is_time_invariant:
            issues.extend(_check_time_coverage(parsed_ts, causal_gran, ind_name))
            issues.extend(_check_timestamp_gaps(parsed_ts, causal_gran, ind_name))

        # 5. Hallucination signals
        issues.extend(
            _check_hallucination_signals(
                values["value"], dtype or "continuous", ind_name
            )
        )

    # 6. Cross-indicator construct correlations
    issues.extend(_check_construct_correlations(combined, indicators))

    errors = [i for i in issues if i["severity"] == "error"]
    is_valid = len(errors) == 0

    return {"is_valid": is_valid, "issues": issues}


@task(cache_policy=INPUTS)
def combine_worker_results(
    worker_results: list["WorkerResult"],
) -> pl.DataFrame:
    """Combine raw worker results into a single DataFrame.

    This produces the raw timestamped data that the SSM will use directly.
    No aggregation is performed - the model handles irregular
    time intervals via continuous-time discretization.

    Args:
        worker_results: List of WorkerResults from Stage 2

    Returns:
        Combined DataFrame with columns: indicator, value, timestamp
    """
    dataframes = [wr.dataframe for wr in worker_results if wr.dataframe is not None]
    if not dataframes:
        return pl.DataFrame({"indicator": [], "value": [], "timestamp": []})

    return pl.concat(dataframes, how="vertical")


@task(cache_policy=INPUTS)
def aggregate_measurements(
    causal_spec: dict,
    worker_results: list["WorkerResult"],
) -> dict[str, pl.DataFrame]:
    """Aggregate raw worker extractions to measurement_granularity.

    Workers extract at the finest resolution visible in their chunk.
    This task buckets timestamps and applies each indicator's aggregation
    function to reach measurement_granularity.

    Args:
        causal_spec: The full causal spec with measurement model
        worker_results: List of WorkerResults from Stage 2

    Returns:
        Dict keyed by granularity level (e.g. "daily", "hourly", "finest").
        Each value is a DataFrame with columns (indicator, value, time_bucket).
    """
    worker_dfs = [wr.dataframe for wr in worker_results]
    return aggregate_worker_measurements(worker_dfs, causal_spec)
