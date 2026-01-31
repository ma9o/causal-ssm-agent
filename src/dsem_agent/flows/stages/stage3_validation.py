"""Stage 3: Transform + Validate (ETL pattern).

This stage performs the "T" (Transform) and validation in an ETL pipeline:
1. **Transform**: Aggregate raw worker extractions into time-bucketed DataFrames
2. **Validate**: Semantic checks that Polars schema can't enforce

This is the gate between extraction (Stage 2) and model specification (Stage 4).

Validation checks (semantic only - Polars handles structural validation):
1. Variance: Indicator has variance > 0 (constant values = zero information)
2. Sample size: Enough time points for temporal modeling

See docs/reference/pipeline.md for full specification.
"""

from typing import TYPE_CHECKING

import polars as pl
from prefect import task
from prefect.cache_policies import INPUTS

from dsem_agent.utils.aggregations import aggregate_worker_measurements

if TYPE_CHECKING:
    from dsem_agent.workers.agents import WorkerResult


# =============================================================================
# TRANSFORM: Aggregate raw extractions into time-bucketed DataFrames
# =============================================================================


@task(cache_policy=INPUTS)
def aggregate_measurements(
    worker_results: list["WorkerResult"],
    dsem_model: dict,
) -> dict[str, pl.DataFrame]:
    """Aggregate raw worker extractions into time-series DataFrames by granularity.

    This is the "T" (Transform) step in the ETL pipeline.

    Takes raw worker outputs (indicator, value, timestamp) and produces
    time-series DataFrames ready for causal modeling:
    1. Concatenates worker DataFrames (indicator, value, timestamp)
    2. Groups indicators by their construct's causal_granularity
    3. Parses timestamps and buckets to each granularity
    4. Applies indicator-specific aggregation (mean, sum, max, etc.)
    5. Returns one DataFrame per granularity with indicators as columns

    Args:
        worker_results: List of WorkerResults from Stage 2 parallel workers
        dsem_model: DSEMModel dict with latent.constructs and measurement.indicators

    Returns:
        Dict mapping granularity -> DataFrame. Each DataFrame has 'time_bucket'
        as first column, then one column per indicator at that granularity.
        Time-invariant indicators (causal_granularity=None) are in key 'time_invariant'
        as a single-row DataFrame.
    """
    dataframes = [wr.dataframe for wr in worker_results]
    return aggregate_worker_measurements(dataframes, dsem_model)


# =============================================================================
# VALIDATE: Semantic checks (variance, sample size)
# =============================================================================

# Minimum observations per granularity for temporal modeling
MIN_OBSERVATIONS = {
    "hourly": 48,  # 2 days of hourly data
    "daily": 14,  # 2 weeks
    "weekly": 8,  # 2 months
    "monthly": 6,  # 6 months
    "yearly": 3,  # 3 years
    "time_invariant": 1,  # Just needs to exist
}


def _get_indicator_granularity(indicator: dict, constructs: list[dict]) -> str | None:
    """Get the causal_granularity for an indicator from its construct."""
    construct_name = indicator.get("construct") or indicator.get("construct_name")
    for c in constructs:
        if c.get("name") == construct_name:
            return c.get("causal_granularity")
    return None


@task(cache_policy=INPUTS)
def validate_extraction(
    dsem_model: dict,
    extracted_data: dict[str, pl.DataFrame],
) -> dict:
    """Validate semantic properties of extracted data.

    Only checks properties that Polars schema can't enforce:
    - Variance > 0 (constant values are uninformative)
    - Sample size (enough time points for temporal modeling)

    Args:
        dsem_model: The full DSEM model with measurement model
        extracted_data: Dict of granularity -> polars DataFrame

    Returns:
        Dict with:
            - is_valid: bool
            - issues: list of {indicator, issue_type, severity, message}
    """
    indicators = dsem_model.get("measurement", {}).get("indicators", [])
    constructs = dsem_model.get("latent", {}).get("constructs", [])

    issues: list[dict] = []

    for indicator in indicators:
        ind_name = indicator.get("name")
        if not ind_name:
            continue

        granularity = _get_indicator_granularity(indicator, constructs)
        gran_key = granularity if granularity else "time_invariant"

        df = extracted_data.get(gran_key)
        if df is None or ind_name not in df.columns:
            # Structural issue - Polars/downstream will handle
            continue

        series = df[ind_name]
        n_non_null = len(series) - series.null_count()

        if n_non_null == 0:
            # No data to check
            continue

        # Check variance
        try:
            variance = series.drop_nulls().var()
            if variance is not None and variance == 0:
                min_val = series.drop_nulls().min()
                issues.append({
                    "indicator": ind_name,
                    "issue_type": "no_variance",
                    "severity": "error",
                    "message": f"Zero variance (constant value = {min_val})",
                })
        except Exception:
            pass

        # Check sample size
        min_n = MIN_OBSERVATIONS.get(gran_key, 10)
        if n_non_null < min_n:
            issues.append({
                "indicator": ind_name,
                "issue_type": "low_n",
                "severity": "warning",
                "message": f"Only {n_non_null} observations (recommend >= {min_n})",
            })

    errors = [i for i in issues if i["severity"] == "error"]
    is_valid = len(errors) == 0

    return {"is_valid": is_valid, "issues": issues}
