"""Aggregation registry for DSEM time-series aggregations using Polars."""

from typing import Callable

import polars as pl

# Type alias for aggregator functions
# Takes column name, returns Polars expression
Aggregator = Callable[[str], pl.Expr]


def agg_mean(col: str) -> pl.Expr:
    """Mean aggregation."""
    return pl.col(col).mean()


def agg_p90(col: str) -> pl.Expr:
    """90th percentile."""
    return pl.col(col).quantile(0.9)


def agg_entropy(col: str) -> pl.Expr:
    """Shannon entropy (normalized counts)."""
    # For continuous data, this computes entropy of binned values
    # For categorical, computes entropy directly
    return (
        pl.col(col)
        .value_counts()
        .struct.field("count")
        .cast(pl.Float64)
        .map_batches(
            lambda s: pl.Series([-((p := s / s.sum()) * (p + 1e-10).log()).sum()])
        )
        .first()
    )


def agg_range(col: str) -> pl.Expr:
    """Range (max - min)."""
    return pl.col(col).max() - pl.col(col).min()


def agg_cv(col: str) -> pl.Expr:
    """Coefficient of variation (std/mean)."""
    return pl.col(col).std() / (pl.col(col).mean() + 1e-10)


def agg_iqr(col: str) -> pl.Expr:
    """Interquartile range (p75 - p25)."""
    return pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)


# Main aggregation registry
AGGREGATION_REGISTRY: dict[str, Aggregator] = {
    # --- Standard statistics ---
    "mean": agg_mean,
    "sum": lambda c: pl.col(c).sum(),
    "min": lambda c: pl.col(c).min(),
    "max": lambda c: pl.col(c).max(),
    "std": lambda c: pl.col(c).std(),
    "var": lambda c: pl.col(c).var(),
    "last": lambda c: pl.col(c).last(),
    "first": lambda c: pl.col(c).first(),
    "count": lambda c: pl.col(c).count(),
    # --- Distributional ---
    "median": lambda c: pl.col(c).median(),
    "p10": lambda c: pl.col(c).quantile(0.1),
    "p25": lambda c: pl.col(c).quantile(0.25),
    "p75": lambda c: pl.col(c).quantile(0.75),
    "p90": agg_p90,
    "p99": lambda c: pl.col(c).quantile(0.99),
    "skew": lambda c: pl.col(c).skew(),
    "kurtosis": lambda c: pl.col(c).kurtosis(),
    "iqr": agg_iqr,
    # --- Spread/variability ---
    "range": agg_range,
    "cv": agg_cv,
    # --- Domain-specific ---
    "entropy": agg_entropy,
    "instability": lambda c: pl.col(c).diff().abs().mean(),  # Mean absolute change
    "trend": lambda c: pl.col(c).diff().mean(),  # Average direction of change
    "n_unique": lambda c: pl.col(c).n_unique(),
}


def get_aggregator(name: str) -> Aggregator:
    """Get an aggregator function by name.

    Args:
        name: Aggregation function name (must be in AGGREGATION_REGISTRY)

    Returns:
        Polars expression factory function

    Raises:
        ValueError: If aggregation name is not in registry
    """
    if name not in AGGREGATION_REGISTRY:
        available = ", ".join(sorted(AGGREGATION_REGISTRY.keys()))
        raise ValueError(f"Unknown aggregation '{name}'. Available: {available}")
    return AGGREGATION_REGISTRY[name]


def list_aggregations() -> list[str]:
    """List all available aggregation names."""
    return sorted(AGGREGATION_REGISTRY.keys())


def apply_aggregation(df: pl.DataFrame, col: str, agg_name: str, group_by: list[str] | None = None) -> pl.DataFrame:
    """Apply an aggregation to a DataFrame column.

    Args:
        df: Input DataFrame
        col: Column to aggregate
        agg_name: Name of aggregation from registry
        group_by: Optional columns to group by before aggregating

    Returns:
        Aggregated DataFrame
    """
    agg_fn = get_aggregator(agg_name)
    expr = agg_fn(col).alias(f"{col}_{agg_name}")

    if group_by:
        return df.group_by(group_by).agg(expr)
    return df.select(expr)


def _truncate_to_granularity(ts: pl.Expr, granularity: str) -> pl.Expr:
    """Truncate a datetime expression to the specified granularity.

    Args:
        ts: Polars datetime expression
        granularity: One of 'hourly', 'daily', 'weekly', 'monthly', 'yearly'

    Returns:
        Truncated datetime expression
    """
    truncate_map = {
        "hourly": "1h",
        "daily": "1d",
        "weekly": "1w",
        "monthly": "1mo",
        "yearly": "1y",
    }
    if granularity not in truncate_map:
        raise ValueError(f"Unknown granularity '{granularity}'. Must be one of: {list(truncate_map.keys())}")
    return ts.dt.truncate(truncate_map[granularity])


def _coerce_value_to_numeric(value) -> float | None:
    """Coerce a value to numeric, returning None if not possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _get_indicator_metadata(dsem_model: dict) -> dict[str, dict]:
    """Extract indicator metadata from a DSEMModel dict.

    Supports both new format (structural + measurement) and old format (dimensions).

    Args:
        dsem_model: DSEMModel dict (new or old format)

    Returns:
        Dict mapping indicator name to {causal_granularity, aggregation}
    """
    # New format: structural + measurement
    if "measurement" in dsem_model and "indicators" in dsem_model["measurement"]:
        indicators = dsem_model["measurement"]["indicators"]
        constructs = dsem_model.get("structural", {}).get("constructs", [])

        # Build construct name -> causal_granularity map
        construct_granularity = {
            c.get("name"): c.get("causal_granularity")
            for c in constructs
        }

        indicator_info = {}
        for ind in indicators:
            name = ind.get("name")
            if not name:
                continue
            # Get causal_granularity from the construct this indicator measures
            construct_name = ind.get("construct") or ind.get("construct_name")
            causal_gran = construct_granularity.get(construct_name)

            indicator_info[name] = {
                "causal_granularity": causal_gran,
                "aggregation": ind.get("aggregation", "mean"),
            }
        return indicator_info

    # Old format: dimensions with observability
    if "dimensions" in dsem_model:
        dim_info = {}
        for dim in dsem_model["dimensions"]:
            name = dim.get("name")
            if not name:
                continue
            # Only process observed dimensions (latent have no measurements)
            if dim.get("observability") == "latent":
                continue
            dim_info[name] = {
                "causal_granularity": dim.get("causal_granularity"),
                "aggregation": dim.get("aggregation", "mean"),
            }
        return dim_info

    return {}


def aggregate_worker_measurements(
    dataframes: list[pl.DataFrame],
    dsem_model: dict,
) -> dict[str, pl.DataFrame]:
    """Aggregate worker measurements into time-series DataFrames by causal_granularity.

    Takes raw worker outputs (indicator, value, timestamp) and produces
    time-series DataFrames ready for causal modeling:
    1. Concatenates all worker DataFrames
    2. Parses timestamps
    3. Groups indicators by their construct's causal_granularity
    4. For each granularity, buckets timestamps and applies aggregation
    5. Returns one DataFrame per granularity with indicators as columns

    Args:
        dataframes: List of DataFrames from workers, each with columns
                   (indicator, value, timestamp). Also supports legacy format
                   with (dimension, value, timestamp).
        dsem_model: DSEMModel dict (new or old format)

    Returns:
        Dict mapping granularity -> DataFrame. Each DataFrame has 'time_bucket'
        as first column, then one column per indicator at that granularity.
        Time-invariant indicators (causal_granularity=None) are in key 'time_invariant'
        as a single-row DataFrame.
    """
    if not dataframes:
        return {}

    # Concatenate all worker dataframes
    combined = pl.concat(dataframes, how="vertical")

    if combined.is_empty():
        return {}

    # Normalize column name: support both "indicator" and "dimension"
    if "dimension" in combined.columns and "indicator" not in combined.columns:
        combined = combined.rename({"dimension": "indicator"})

    # Build indicator metadata from dsem_model
    ind_info = _get_indicator_metadata(dsem_model)

    # Parse timestamps - try multiple formats
    # time_zone="UTC" handles timestamps with timezone info (e.g., +00:00 or Z suffix)
    combined = combined.with_columns(
        pl.col("timestamp")
        .str.to_datetime(strict=False, time_zone="UTC")
        .alias("parsed_ts")
    )

    # Coerce values to numeric
    combined = combined.with_columns(
        pl.col("value")
        .map_elements(_coerce_value_to_numeric, return_dtype=pl.Float64)
        .alias("numeric_value")
    )

    # Group indicators by granularity
    inds_by_granularity: dict[str | None, list[str]] = {}
    for ind_name, info in ind_info.items():
        gran = info["causal_granularity"]
        if gran not in inds_by_granularity:
            inds_by_granularity[gran] = []
        inds_by_granularity[gran].append(ind_name)

    results: dict[str, pl.DataFrame] = {}

    for granularity, ind_names in inds_by_granularity.items():
        if granularity is None:
            # Time-invariant indicators - aggregate all values into single row
            time_invariant_cols = {}
            for ind_name in ind_names:
                ind_data = combined.filter(pl.col("indicator") == ind_name)
                if ind_data.is_empty():
                    continue

                agg_name = ind_info[ind_name]["aggregation"]
                try:
                    agg_fn = get_aggregator(agg_name)
                except ValueError:
                    agg_fn = get_aggregator("mean")

                agg_value = ind_data.select(agg_fn("numeric_value")).item()
                time_invariant_cols[ind_name] = [agg_value]

            if time_invariant_cols:
                results["time_invariant"] = pl.DataFrame(time_invariant_cols)
            continue

        # Time-varying indicators at this granularity
        # Filter to rows with valid timestamps
        gran_data = combined.filter(
            pl.col("parsed_ts").is_not_null() &
            pl.col("indicator").is_in(ind_names)
        )

        if gran_data.is_empty():
            continue

        # Process each indicator and collect for joining
        ind_dfs = []
        for ind_name in ind_names:
            ind_data = gran_data.filter(pl.col("indicator") == ind_name)
            if ind_data.is_empty():
                continue

            # Bucket timestamps to this granularity
            ind_data = ind_data.with_columns(
                _truncate_to_granularity(pl.col("parsed_ts"), granularity).alias("time_bucket")
            )

            # Get aggregation function
            agg_name = ind_info[ind_name]["aggregation"]
            try:
                agg_fn = get_aggregator(agg_name)
            except ValueError:
                agg_fn = get_aggregator("mean")

            # Group by time bucket and aggregate
            aggregated = (
                ind_data
                .group_by("time_bucket")
                .agg(agg_fn("numeric_value").alias(ind_name))
                .sort("time_bucket")
            )

            ind_dfs.append(aggregated)

        if not ind_dfs:
            continue

        # Join all indicators at this granularity on time_bucket
        result = ind_dfs[0]
        for df in ind_dfs[1:]:
            result = result.join(df, on="time_bucket", how="full", coalesce=True)

        # Sort by time
        result = result.sort("time_bucket")
        results[granularity] = result

    return results
