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
