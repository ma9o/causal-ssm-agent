"""Aggregate raw worker extractions to measurement_granularity.

Workers extract at the finest resolution visible in their chunk.
This module buckets timestamps and applies each indicator's aggregation
function to reach measurement_granularity. SSM discretization then
handles measurement_granularity -> continuous time.

    Workers extract at raw resolution
      -> aggregate_worker_measurements() -> measurement_granularity
      -> SSM discretization -> continuous time
"""

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Granularity -> Polars truncation interval
_TRUNCATE_INTERVAL = {
    "hourly": "1h",
    "daily": "1d",
    "weekly": "1w",
    "monthly": "1mo",
    "yearly": "1y",
}

# Aggregations that require map_groups (cannot be expressed as a single Polars expr)
_MAP_GROUPS_AGGREGATIONS = {"trend"}


def _build_agg_expr(agg_name: str) -> pl.Expr:
    """Map an aggregation name to a Polars expression over the 'value' column.

    Supports 23 of 24 aggregation functions as expressions. The 'trend'
    aggregation requires map_groups and is handled separately in
    aggregate_worker_measurements.
    """
    col = pl.col("value")

    simple = {
        "mean": col.mean(),
        "sum": col.sum(),
        "min": col.min(),
        "max": col.max(),
        "std": col.std(),
        "var": col.var(),
        "last": col.last(),
        "first": col.first(),
        "count": col.count(),
        "median": col.median(),
        "n_unique": col.n_unique(),
        "skew": col.skew(),
        "kurtosis": col.kurtosis(),
        "entropy": col.entropy(),
    }

    if agg_name in simple:
        return simple[agg_name].alias("value")

    # Percentiles
    percentiles = {
        "p10": 0.10,
        "p25": 0.25,
        "p75": 0.75,
        "p90": 0.90,
        "p99": 0.99,
    }
    if agg_name in percentiles:
        q = percentiles[agg_name]
        return col.quantile(q).alias("value")

    # Composite aggregations
    if agg_name == "range":
        return (col.max() - col.min()).alias("value")

    if agg_name == "iqr":
        return (col.quantile(0.75) - col.quantile(0.25)).alias("value")

    if agg_name == "cv":
        return (col.std() / col.mean()).alias("value")

    # MSSD: mean squared successive differences
    if agg_name == "instability":
        return (col.diff().pow(2).mean()).alias("value")

    raise ValueError(f"Unknown aggregation function: '{agg_name}'")


def _build_map_groups_fn(agg_name: str):
    """Return a callable for use with group_by().map_groups().

    Used for aggregations that cannot be expressed as a single Polars expression.
    """
    if agg_name == "trend":

        def _ols_slope(df: pl.DataFrame) -> pl.DataFrame:
            values = df["value"].to_numpy()
            n = len(values)
            if n < 2:
                slope = 0.0
            else:
                x = np.arange(n, dtype=np.float64)
                slope = float(np.polyfit(x, values, 1)[0])
            return df.head(1).with_columns(pl.lit(slope).alias("value"))

        return _ols_slope

    raise ValueError(f"Unknown map_groups aggregation: '{agg_name}'")


_BINARY_TRUE = {"true", "yes", "1", "1.0", "t", "y"}
_BINARY_FALSE = {"false", "no", "0", "0.0", "f", "n"}


def _encode_non_continuous(
    df: pl.DataFrame,
    dtype_lookup: dict[str, str],
    ordinal_levels_lookup: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """Encode non-continuous indicator values to numeric before Float64 cast.

    - binary: map true/false/yes/no/1/0 → 1.0/0.0
    - ordinal: integer label-encode using ordinal_levels order (or sorted fallback)
    - categorical: integer label-encode (sorted categories)
    - continuous/count: no-op (already numeric)

    Modifies the 'value' column in-place per indicator partition.
    """
    if not dtype_lookup:
        return df

    ordinal_levels_lookup = ordinal_levels_lookup or {}

    non_continuous = {
        name: dtype
        for name, dtype in dtype_lookup.items()
        if dtype in ("binary", "ordinal", "categorical")
    }
    if not non_continuous:
        return df

    # Ensure value is Utf8 for string matching
    if df.schema.get("value") != pl.Utf8:
        df = df.with_columns(pl.col("value").cast(pl.Utf8, strict=False))

    frames = []
    remaining_mask = pl.lit(True)

    for name, dtype in non_continuous.items():
        indicator_mask = pl.col("indicator") == name
        subset = df.filter(indicator_mask)
        if subset.is_empty():
            continue

        remaining_mask = remaining_mask & ~indicator_mask

        if dtype == "binary":
            subset = subset.with_columns(
                pl.col("value")
                .str.to_lowercase()
                .map_elements(
                    lambda v: 1.0 if v in _BINARY_TRUE else (0.0 if v in _BINARY_FALSE else None),
                    return_dtype=pl.Float64,
                )
                .alias("value")
            )
            n_null = subset["value"].null_count()
            if n_null > 0:
                logger.warning(
                    "Binary indicator '%s': %d/%d values could not be encoded",
                    name,
                    n_null,
                    len(subset),
                )
        else:
            # ordinal/categorical: label encoding
            # Use explicit ordinal_levels if provided, otherwise fall back to sorted
            explicit_levels = ordinal_levels_lookup.get(name)
            if explicit_levels and dtype == "ordinal":
                unique_vals = explicit_levels
            else:
                unique_vals = sorted(
                    v for v in subset["value"].unique().to_list() if v is not None
                )
            label_map = {v: float(i) for i, v in enumerate(unique_vals)}
            subset = subset.with_columns(
                pl.col("value")
                .map_elements(lambda v, _lm=label_map: _lm.get(v), return_dtype=pl.Float64)
                .alias("value")
            )
            logger.info(
                "%s indicator '%s': label-encoded %d categories",
                dtype.capitalize(),
                name,
                len(unique_vals),
            )

        # Cast value back to Utf8 for consistency with remaining data
        subset = subset.with_columns(pl.col("value").cast(pl.Utf8, strict=False))
        frames.append(subset)

    if not frames:
        return df

    remaining = df.filter(remaining_mask)
    return pl.concat([remaining, *frames], how="vertical")


def aggregate_worker_measurements(
    worker_dfs: list[pl.DataFrame | None],
    causal_spec: dict,
) -> dict[str, pl.DataFrame]:
    """Aggregate raw worker extractions to measurement_granularity.

    Args:
        worker_dfs: List of DataFrames with columns (indicator, value, timestamp),
                    each from a worker. None entries are skipped.
        causal_spec: The full CausalSpec dict with measurement.indicators.

    Returns:
        Dict keyed by granularity level (e.g. "daily", "hourly", "finest").
        Each value is a DataFrame with columns (indicator, value, time_bucket).
        For "finest" granularity, time_bucket is the original datetime.
    """
    # Filter out None DataFrames and empty list
    valid_dfs = [df for df in worker_dfs if df is not None and not df.is_empty()]
    if not valid_dfs:
        return {}

    # 1. Concat all worker DataFrames
    combined = pl.concat(valid_dfs, how="vertical")

    # 2. Materialize Object dtype columns to Utf8 for further processing
    for col_name in ("value", "timestamp"):
        if col_name in combined.columns and combined.schema[col_name] == pl.Object:
            py_vals = [str(v) if v is not None else None for v in combined[col_name].to_list()]
            combined = combined.with_columns(pl.Series(col_name, py_vals, dtype=pl.Utf8))

    # 3. Dtype-aware encoding: encode binary/ordinal/categorical values before Float64 cast
    indicators = causal_spec.get("measurement", {}).get("indicators", [])
    dtype_lookup = {
        ind.get("name"): ind.get("measurement_dtype", "continuous") for ind in indicators
    }
    ordinal_levels_lookup = {
        ind.get("name"): ind["ordinal_levels"]
        for ind in indicators
        if ind.get("ordinal_levels")
    }
    combined = _encode_non_continuous(combined, dtype_lookup, ordinal_levels_lookup)

    combined = combined.with_columns(
        pl.col("value").cast(pl.Float64, strict=False).alias("value"),
        pl.col("timestamp")
        .cast(pl.Utf8, strict=False)
        .str.to_datetime(strict=False)
        .alias("datetime"),
    )

    # 4. Drop rows with null timestamp or null value
    combined = combined.drop_nulls(subset=["datetime", "value"])

    if combined.is_empty():
        return {}

    # 5. Build indicator -> (measurement_granularity, aggregation) lookup
    indicators = causal_spec.get("measurement", {}).get("indicators", [])
    indicator_info: dict[str, dict[str, str]] = {}
    for ind in indicators:
        name = ind.get("name")
        if name:
            indicator_info[name] = {
                "granularity": ind.get("measurement_granularity", "finest"),
                "aggregation": ind.get("aggregation", "mean"),
            }

    # 6. Filter to known indicators only
    known_names = set(indicator_info.keys())
    combined = combined.filter(pl.col("indicator").is_in(known_names))

    if combined.is_empty():
        return {}

    # 7. Group indicators by their measurement_granularity
    granularity_groups: dict[str, list[str]] = {}
    for name, info in indicator_info.items():
        gran = info["granularity"]
        granularity_groups.setdefault(gran, []).append(name)

    results: dict[str, pl.DataFrame] = {}

    for gran, ind_names in granularity_groups.items():
        subset = combined.filter(pl.col("indicator").is_in(ind_names))
        if subset.is_empty():
            continue

        if gran == "finest":
            # Deduplicate exact (indicator, datetime, value) triples
            deduped = subset.unique(subset=["indicator", "datetime", "value"])
            results["finest"] = deduped.select(
                pl.col("indicator"),
                pl.col("value"),
                pl.col("datetime").alias("time_bucket"),
            ).sort("indicator", "time_bucket")
        else:
            # Bucket timestamps and aggregate
            interval = _TRUNCATE_INTERVAL.get(gran)
            if interval is None:
                continue

            # Add time_bucket column
            bucketed = subset.with_columns(
                pl.col("datetime").dt.truncate(interval).alias("time_bucket"),
            )

            # Sort by datetime so first/last are chronological
            bucketed = bucketed.sort("datetime")

            # Aggregate per indicator per time_bucket
            agg_frames = []
            for ind_name in ind_names:
                ind_data = bucketed.filter(pl.col("indicator") == ind_name)
                if ind_data.is_empty():
                    continue

                agg_name = indicator_info[ind_name]["aggregation"]

                if agg_name in _MAP_GROUPS_AGGREGATIONS:
                    fn = _build_map_groups_fn(agg_name)
                    agged = (
                        ind_data.group_by("time_bucket", maintain_order=True)
                        .map_groups(fn)
                        .with_columns(pl.lit(ind_name).alias("indicator"))
                        .select("indicator", "value", "time_bucket")
                    )
                else:
                    agg_expr = _build_agg_expr(agg_name)
                    agged = (
                        ind_data.group_by("time_bucket", maintain_order=True)
                        .agg(agg_expr)
                        .with_columns(pl.lit(ind_name).alias("indicator"))
                        .select("indicator", "value", "time_bucket")
                    )
                agg_frames.append(agged)

            if agg_frames:
                results[gran] = pl.concat(agg_frames, how="vertical").sort(
                    "indicator", "time_bucket"
                )

    return results


def flatten_aggregated_data(aggregated: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Flatten dict[granularity -> DataFrame] to single long-format DataFrame.

    Args:
        aggregated: Dict keyed by granularity level, each with
            columns (indicator, value, time_bucket).

    Returns:
        Single DataFrame with columns (indicator, value, time_bucket),
        sorted by indicator then time_bucket.
    """
    frames = list(aggregated.values())
    if not frames:
        return pl.DataFrame({"indicator": [], "value": [], "time_bucket": []})
    return pl.concat(frames, how="vertical").sort("indicator", "time_bucket")
