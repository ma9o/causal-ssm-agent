"""Pure helper functions used by both the Hamilton DAG nodes and the pipeline.

No Prefect or Hamilton imports here — just data transformations.
"""

from __future__ import annotations

import polars as pl

from causal_ssm_agent.utils.causal_spec import get_indicators


def format_schema_for_llm(df: pl.DataFrame, column_descriptions: dict[str, str]) -> str:
    """Format a DataFrame schema and sample for LLM consumption.

    Used by Stage 1b so the LLM can see what columns are available
    when proposing the measurement model.
    """
    lines = ["## Dataset Schema\n"]
    lines.append("| Column | Type | Description |")
    lines.append("|--------|------|-------------|")
    for col in df.columns:
        dtype = str(df.schema[col])
        desc = column_descriptions.get(col, "")
        lines.append(f"| {col} | {dtype} | {desc} |")

    lines.append("\n## Sample Data (first 10 rows)\n")
    lines.append(str(df.head(10)))

    lines.append("\n## Summary\n")
    lines.append(f"- Total rows: {len(df)}")
    lines.append(f"- Total columns: {len(df.columns)}")

    # Basic stats for numeric columns
    numeric_cols = [c for c in df.columns if df.schema[c].is_numeric()]
    if numeric_cols:
        lines.append("\n## Numeric Column Statistics\n")
        lines.append(str(df.select(numeric_cols).describe()))

    return "\n".join(lines)


def map_columns_to_indicators(df: pl.DataFrame, causal_spec: dict) -> pl.DataFrame:
    """Map ingested DataFrame columns to the long-format indicator representation.

    Takes a wide-format DataFrame (one column per variable) and melts it to the
    long format (indicator, value, timestamp) expected by Stage 3+.
    """
    indicators = get_indicators(causal_spec)
    indicator_names = [ind["name"] for ind in indicators]

    available = [name for name in indicator_names if name in df.columns]
    if not available:
        raise ValueError(
            f"No indicator columns found in DataFrame. "
            f"Expected: {indicator_names}, got: {df.columns}"
        )

    # Detect timestamp column
    time_col = None
    for candidate in ("timestamp", "date", "time", "datetime", "time_bucket"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        for col in df.columns:
            if df.schema[col] in (pl.Date, pl.Datetime):
                time_col = col
                break

    if time_col:
        long_df = df.select([time_col, *available]).unpivot(
            index=time_col,
            on=available,
            variable_name="indicator",
            value_name="value",
        )
        if time_col != "timestamp":
            long_df = long_df.rename({time_col: "timestamp"})
        long_df = long_df.with_columns(
            pl.col("timestamp").cast(pl.Utf8),
            pl.col("value").cast(pl.Utf8),
        )
    else:
        long_df = df.select(available).unpivot(
            on=available,
            variable_name="indicator",
            value_name="value",
        )
        long_df = long_df.with_columns(
            pl.lit(None).alias("timestamp"),
            pl.col("value").cast(pl.Utf8),
        )

    long_df = long_df.drop_nulls(subset=["value"])
    return long_df


def compute_date_range(df: pl.DataFrame) -> dict[str, str]:
    """Compute date range from a DataFrame's time-like columns."""
    for candidate in ("timestamp", "date", "time", "datetime"):
        if candidate in df.columns:
            col = df[candidate]
            if col.dtype in (pl.Date, pl.Datetime):
                start = col.min()
                end = col.max()
                if start is not None and end is not None:
                    return {"start": str(start)[:10], "end": str(end)[:10]}
            elif col.dtype == pl.Utf8:
                try:
                    parsed = col.str.to_datetime(strict=False).drop_nulls()
                    if len(parsed) > 0:
                        return {
                            "start": str(parsed.min())[:10],
                            "end": str(parsed.max())[:10],
                        }
                except Exception:
                    pass
    return {"start": "", "end": ""}


def sample_rows(df: pl.DataFrame, n: int = 15) -> list[dict[str, str | None]]:
    """Sample rows from a DataFrame for web display."""
    if df.is_empty():
        return []
    total = len(df)
    if total <= n:
        sample = df
    else:
        step = (total - 1) / (n - 1)
        indices = [round(i * step) for i in range(n)]
        sample = df[indices]
    rows = []
    for row_dict in sample.to_dicts():
        rows.append({k: (str(v) if v is not None else None) for k, v in row_dict.items()})
    return rows


def build_stage0_payload(ingestion_result: object, df: pl.DataFrame) -> dict:
    """Build the web-serializable stage 0 payload from an IngestionResult."""
    return {
        "source_label": ingestion_result.source_label,  # type: ignore[attr-defined]
        "n_records": df.shape[0],
        "n_columns": df.shape[1],
        "date_range": compute_date_range(df),
        "sample": sample_rows(df),
        "column_descriptions": [
            {
                "name": col,
                "dtype": str(df.schema[col]),
                "description": desc,
            }
            for col, desc in ingestion_result.column_descriptions.items()  # type: ignore[attr-defined]
        ],
        "llm_trace": ingestion_result.llm_trace,  # type: ignore[attr-defined]
    }
