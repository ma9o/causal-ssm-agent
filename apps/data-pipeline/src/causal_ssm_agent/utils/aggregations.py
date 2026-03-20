"""Dtype encoding and aggregation helpers for extracted indicator data.

Provides non-continuous dtype encoding (binary, ordinal, categorical -> numeric)
and Polars aggregation expression builders used by the pipeline's stage 2 logic.
"""

import numpy as np
import polars as pl

from causal_ssm_agent.flows import get_prefect_logger

logger = get_prefect_logger(__name__)

# Aggregations that require map_groups (cannot be expressed as a single Polars expr)
_MAP_GROUPS_AGGREGATIONS = {"trend"}


def _build_agg_expr(agg_name: str, col_name: str = "value") -> pl.Expr:
    """Map an aggregation name to a Polars expression over a named column.

    Supports 23 of 24 aggregation functions as expressions. The 'trend'
    aggregation requires map_groups and is handled separately via
    _build_map_groups_fn.

    Args:
        agg_name: Name of the aggregation function.
        col_name: Column to aggregate (default: "value").

    Returns:
        Polars expression aliased to "value".
    """
    col = pl.col(col_name)

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
        return (
            pl.when(col.mean().abs() > 1e-15).then(col.std() / col.mean()).otherwise(None)
        ).alias("value")

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


def compute_indicators(
    raw_df: pl.DataFrame,
    indicators: list[dict],
    model_clock: str,
    time_col: str,
) -> pl.DataFrame:
    """Compute indicator values directly via Polars aggregation.

    For indicators with extraction_mode='computed', applies the aggregation
    function to the single source column, grouped by each indicator's effective
    observation window (explicit observation_window or fallback model_clock).

    Args:
        raw_df: Raw wide-format DataFrame with actual column names.
        indicators: List of indicator dicts with extraction_mode="computed".
            Each must have exactly one source_column.
        model_clock: Global fallback duration string for truncation (e.g., "1d").
        time_col: Name of the datetime column in raw_df.

    Returns:
        Long-format DataFrame with columns: indicator (Utf8), value (Utf8),
        timestamp (Utf8). Matches the schema produced by the semantic path.
    """
    output_schema = {"indicator": pl.Utf8, "value": pl.Utf8, "timestamp": pl.Utf8}
    if not indicators:
        return pl.DataFrame(schema=output_schema)

    # Match the rest of the pipeline: raw string timestamps may already carry
    # a timezone suffix such as `Z`, so parse them as UTC.
    df = raw_df
    if df.schema[time_col] == pl.Utf8:
        df = df.with_columns(
            pl.col(time_col).str.to_datetime(strict=False, time_zone="UTC").alias(time_col)
        )

    frames: list[pl.DataFrame] = []
    for ind in indicators:
        name = ind["name"]
        source_col = ind["source_columns"][0]
        agg_name = ind["aggregation"]
        observation_window = ind.get("observation_window") or model_clock

        if source_col not in df.columns:
            logger.warning(
                "Computed indicator '%s': source column '%s' not in DataFrame, skipping",
                name,
                source_col,
            )
            continue

        if agg_name in _MAP_GROUPS_AGGREGATIONS:
            # trend etc: rename source_col → "value" for map_groups function
            fn = _build_map_groups_fn(agg_name)
            agg_df = (
                df.select(
                    pl.col(time_col).dt.truncate(observation_window).alias("__tick__"),
                    pl.col(source_col).cast(pl.Float64, strict=False).alias("value"),
                )
                .sort("__tick__")
                .group_by("__tick__", maintain_order=True)
                .map_groups(fn)
            )
        else:
            expr = _build_agg_expr(agg_name, source_col)
            agg_df = (
                df.select(
                    pl.col(time_col).dt.truncate(observation_window).alias("__tick__"),
                    pl.col(source_col).cast(pl.Float64, strict=False).alias(source_col),
                )
                .group_by("__tick__", maintain_order=True)
                .agg(expr)
            )

        agg_df = agg_df.select(
            pl.lit(name).alias("indicator"),
            pl.col("value").cast(pl.Utf8).alias("value"),
            pl.col("__tick__").dt.to_string("%Y-%m-%dT%H:%M:%S").alias("timestamp"),
        )
        frames.append(agg_df)

    if not frames:
        return pl.DataFrame(schema=output_schema)

    return pl.concat(frames, how="vertical").sort("timestamp", "indicator")


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
                unique_vals = sorted(v for v in subset["value"].unique().to_list() if v is not None)
            # Normalize for case-insensitive matching (mirrors binary branch)
            label_map = {
                v.strip().lower() if isinstance(v, str) else v: float(i)
                for i, v in enumerate(unique_vals)
            }
            subset = subset.with_columns(
                pl.col("value")
                .str.strip_chars()
                .str.to_lowercase()
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
