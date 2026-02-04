"""Stage 3: Validate extracted data.

Validation checks (semantic only - Polars handles structural validation):
1. Variance: Indicator has variance > 0 (constant values = zero information)
2. Sample size: Enough observations for temporal modeling

NOTE: Upfront aggregation has been removed. The CT-SEM model handles
raw timestamped data directly, discretizing at inference time based on
actual observation intervals.

See docs/reference/pipeline.md for full specification.
"""

from typing import TYPE_CHECKING

import polars as pl
from prefect import task
from prefect.cache_policies import INPUTS

if TYPE_CHECKING:
    from dsem_agent.workers.agents import WorkerResult


# Minimum observations for temporal modeling
MIN_OBSERVATIONS = 10  # Reasonable minimum for CT-SEM


@task(cache_policy=INPUTS)
def validate_extraction(
    dsem_model: dict,
    worker_results: list["WorkerResult"],
) -> dict:
    """Validate semantic properties of extracted data.

    Checks raw worker extractions for:
    - Variance > 0 (constant values are uninformative)
    - Sample size (enough observations for temporal modeling)

    Args:
        dsem_model: The full DSEM model with measurement model
        worker_results: List of WorkerResults from Stage 2

    Returns:
        Dict with:
            - is_valid: bool
            - issues: list of {indicator, issue_type, severity, message}
    """
    # Concatenate all worker dataframes
    dataframes = [wr.dataframe for wr in worker_results if wr.dataframe is not None]
    if not dataframes:
        return {"is_valid": False, "issues": [{"indicator": "all", "issue_type": "no_data", "severity": "error", "message": "No data extracted"}]}

    combined = pl.concat(dataframes, how="vertical")

    if combined.is_empty():
        return {"is_valid": False, "issues": [{"indicator": "all", "issue_type": "no_data", "severity": "error", "message": "No data extracted"}]}

    indicators = dsem_model.get("measurement", {}).get("indicators", [])
    indicator_names = {ind.get("name") for ind in indicators if ind.get("name")}

    issues: list[dict] = []

    # Check each indicator in the extracted data
    for ind_name in indicator_names:
        ind_data = combined.filter(pl.col("indicator") == ind_name)

        if ind_data.is_empty():
            issues.append({
                "indicator": ind_name,
                "issue_type": "missing",
                "severity": "warning",
                "message": "No data extracted for this indicator",
            })
            continue

        # Coerce values to numeric for validation
        values = ind_data.select(
            pl.col("value").cast(pl.Float64, strict=False)
        ).drop_nulls()

        n_obs = len(values)

        if n_obs == 0:
            issues.append({
                "indicator": ind_name,
                "issue_type": "no_numeric",
                "severity": "error",
                "message": "No numeric values extracted",
            })
            continue

        # Check sample size
        if n_obs < MIN_OBSERVATIONS:
            issues.append({
                "indicator": ind_name,
                "issue_type": "low_n",
                "severity": "warning",
                "message": f"Only {n_obs} observations (recommend >= {MIN_OBSERVATIONS})",
            })

        # Check variance
        try:
            variance = values["value"].var()
            if variance is not None and variance == 0:
                const_val = values["value"].first()
                issues.append({
                    "indicator": ind_name,
                    "issue_type": "no_variance",
                    "severity": "error",
                    "message": f"Zero variance (constant value = {const_val})",
                })
        except Exception:
            pass

    errors = [i for i in issues if i["severity"] == "error"]
    is_valid = len(errors) == 0

    return {"is_valid": is_valid, "issues": issues}


@task(cache_policy=INPUTS)
def combine_worker_results(
    worker_results: list["WorkerResult"],
) -> pl.DataFrame:
    """Combine raw worker results into a single DataFrame.

    This produces the raw timestamped data that CT-SEM will use directly.
    No aggregation is performed - the CT-SEM model handles irregular
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
