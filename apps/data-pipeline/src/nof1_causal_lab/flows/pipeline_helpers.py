"""Pure helper functions shared by stage helpers and the pipeline.

No orchestration imports here, just data transformations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


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


def build_stage0_payload(ingestion_result: object) -> dict:
    """Build the web-serializable stage 0 payload from an IngestionResult."""
    return {
        "column_descriptions": [
            {
                "name": col,
                "description": desc,
            }
            for col, desc in ingestion_result.column_descriptions.items()  # type: ignore[attr-defined]
        ],
        "llm_trace": ingestion_result.llm_trace,  # type: ignore[attr-defined]
    }
