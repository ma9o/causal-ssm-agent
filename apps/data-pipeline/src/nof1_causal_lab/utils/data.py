import logging
from typing import TypedDict

import polars as pl

from nof1_causal_lab.utils.causal_design import (
    get_effective_observation_window,
)
from nof1_causal_lab.utils.observation_semantics import (
    AnchorPolicy,
    get_observation_semantics,
)
from nof1_causal_lab.utils.storage import get_base_uri, join

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0


class ObservationRecord(TypedDict):
    """Canonical serialized Stage 2 observation row."""

    indicator: str
    value: str | int | float | bool | None
    anchor_time: str | None
    support_kind: str | None
    summary_operator: str | None
    anchor_policy: str | None
    observation_window: str | None
    support_start: str | None
    support_end: str | None


OBSERVATION_ROW_SCHEMA = {
    "indicator": pl.Utf8,
    "value": pl.Utf8,
    "anchor_time": pl.Utf8,
    "support_kind": pl.Utf8,
    "summary_operator": pl.Utf8,
    "anchor_policy": pl.Utf8,
    "observation_window": pl.Utf8,
    "support_start": pl.Utf8,
    "support_end": pl.Utf8,
}

# Remote-aware base URI (``/abs/path/to/data`` locally, ``s3://bucket/prefix`` on R2)
DATA_URI = get_base_uri()


def input_dir(workspace_id: str) -> str:
    """Return the input directory for a user ID: ``data/{workspace_id}/input/``."""
    return join(DATA_URI, workspace_id, "input")


def runs_dir(workspace_id: str) -> str:
    """Return the single run directory for a user ID: ``data/{workspace_id}/run/``."""
    return join(DATA_URI, workspace_id, "run")


def ensure_datetime_column(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    """Parse a timestamp column to Polars datetime when it arrives as text."""
    if df.schema[time_col] == pl.Utf8:
        return df.with_columns(
            pl.col(time_col).str.to_datetime(strict=False, time_zone="UTC").alias(time_col)
        )
    return df


def support_window_tick_frame(
    df: pl.DataFrame,
    model_clock: str,
    time_col: str,
) -> pl.DataFrame:
    """Materialize every support-window tick spanning the observed raw data."""
    if df.is_empty():
        return pl.DataFrame(schema={"__tick__": pl.Datetime})

    df = ensure_datetime_column(df, time_col)
    observed_ticks = df.select(pl.col(time_col).dt.truncate(model_clock).alias("__tick__"))
    start, end = observed_ticks.select(
        pl.col("__tick__").min().alias("start"),
        pl.col("__tick__").max().alias("end"),
    ).row(0)
    if start is None or end is None:
        return pl.DataFrame(schema={"__tick__": observed_ticks.schema["__tick__"]})

    ticks = pl.datetime_range(start, end, interval=model_clock, eager=True).alias("__tick__")
    return pl.DataFrame({"__tick__": ticks})


def bucket_by_clock(
    df: pl.DataFrame,
    model_clock: str,
    time_col: str,
) -> list[tuple[str, pl.DataFrame]]:
    """Group DataFrame rows by model_clock ticks.

    Args:
        df: Raw DataFrame with a time column.
        model_clock: Polars duration string (e.g. "1d", "4h", "1w").
        time_col: Name of the datetime column to bucket by.

    Returns:
        List of (tick_id, events_df) sorted chronologically, including empty
        support windows between the first and last observed ticks.
        tick_id is an ISO-format string of the tick start time.
    """
    df = ensure_datetime_column(df, time_col)

    # Truncate to tick boundaries
    bucketed = df.with_columns(pl.col(time_col).dt.truncate(model_clock).alias("__tick__")).sort(
        time_col
    )
    tick_frame = support_window_tick_frame(df, model_clock, time_col)

    groups = {
        tick_val[0]: group_df.drop("__tick__")
        for tick_val, group_df in bucketed.group_by("__tick__", maintain_order=True)
    }
    empty_events = df.head(0)
    result = []
    for tick_dt in tick_frame["__tick__"].to_list():
        events = groups.get(tick_dt, empty_events)
        tick_id = tick_dt.isoformat() if hasattr(tick_dt, "isoformat") else str(tick_dt)
        result.append((tick_id, events))

    return result


def observation_row_schema() -> dict[str, pl.DataType | type[pl.DataType]]:
    """Schema for canonical long-format observation rows."""
    return dict(OBSERVATION_ROW_SCHEMA)


def annotate_observation_rows(
    df: pl.DataFrame,
    measurement_structure: dict,
    *,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Attach observation metadata to long-format Stage 2 rows.

    The input ``time_col`` is the support-window start emitted by the computed
    and semantic extraction paths. The canonical observation-row contract keeps:
    - ``anchor_time``: latent-grid attachment time for the observation
    - ``support_start`` / ``support_end``: realized support bounds

    Canonical support semantics are always derived from the measurement structure,
    not preserved from any caller-supplied row metadata.
    """
    if df.is_empty():
        return pl.DataFrame(schema=observation_row_schema())
    for col_name, dtype in OBSERVATION_ROW_SCHEMA.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

    if "indicator" not in df.columns:
        return df

    model_clock = measurement_structure.get("model_clock")
    indicator_rows = []
    for ind in measurement_structure.get("indicators", []):
        if not ind.get("name"):
            continue
        semantics = get_observation_semantics(ind)
        indicator_rows.append(
            {
                "indicator": ind["name"],
                "support_kind_meta": semantics.support_kind.value,
                "summary_operator_meta": semantics.summary_operator.value,
                "anchor_policy_meta": semantics.anchor_policy.value,
                "observation_window_meta": get_effective_observation_window(ind, model_clock),
            }
        )
    kind_df = (
        pl.DataFrame(
            indicator_rows,
            schema={
                "indicator": pl.Utf8,
                "support_kind_meta": pl.Utf8,
                "summary_operator_meta": pl.Utf8,
                "anchor_policy_meta": pl.Utf8,
                "observation_window_meta": pl.Utf8,
            },
        )
        if indicator_rows
        else pl.DataFrame(
            schema={
                "indicator": pl.Utf8,
                "support_kind_meta": pl.Utf8,
                "summary_operator_meta": pl.Utf8,
                "anchor_policy_meta": pl.Utf8,
                "observation_window_meta": pl.Utf8,
            }
        )
    )

    if kind_df.height > 0:
        df = df.join(kind_df, on="indicator", how="left")
    else:
        df = df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("support_kind_meta"),
            pl.lit(None, dtype=pl.Utf8).alias("summary_operator_meta"),
            pl.lit(None, dtype=pl.Utf8).alias("anchor_policy_meta"),
            pl.lit(None, dtype=pl.Utf8).alias("observation_window_meta"),
        )

    ts_expr = (
        # Stage 2 merges support-window starts coming from both paths:
        # computed rows use naive bucket strings while semantic rows can carry
        # the same UTC boundary with an explicit `+00:00` suffix from the
        # worker header. Normalize the redundant UTC suffix so mixed batches
        # parse consistently into the same support bounds.
        pl.col(time_col)
        .str.replace(r"[Zz]$", "")
        .str.replace(r"[+-]\d{2}:\d{2}$", "")
        .str.to_datetime(strict=False)
        if df.schema.get(time_col) == pl.Utf8
        else pl.col(time_col)
    )
    support_start_expr = ts_expr.dt.to_string("%Y-%m-%dT%H:%M:%S")
    observation_window_expr = pl.col("observation_window_meta")
    support_kind_expr = pl.col("support_kind_meta")
    summary_operator_expr = pl.col("summary_operator_meta")
    anchor_policy_expr = pl.col("anchor_policy_meta")
    support_end_expr = (
        pl.when(observation_window_expr.is_not_null())
        .then(ts_expr.dt.offset_by(observation_window_expr).dt.to_string("%Y-%m-%dT%H:%M:%S"))
        .otherwise(support_start_expr)
    )
    anchor_time_expr = (
        pl.when(anchor_policy_expr == AnchorPolicy.SUPPORT_START.value)
        .then(support_start_expr)
        .otherwise(support_end_expr)
    )

    df = df.with_columns(
        anchor_time_expr.alias("anchor_time"),
        support_kind_expr.alias("support_kind"),
        summary_operator_expr.alias("summary_operator"),
        anchor_policy_expr.alias("anchor_policy"),
        observation_window_expr.alias("observation_window"),
        support_start_expr.alias("support_start"),
        support_end_expr.alias("support_end"),
    ).drop(
        "support_kind_meta",
        "summary_operator_meta",
        "anchor_policy_meta",
        "observation_window_meta",
    )

    if time_col != "anchor_time" and time_col in df.columns:
        df = df.drop(time_col)

    return df


def pivot_to_wide(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot long-format observation data to wide-format Polars DataFrame.

    Handles time column detection, Float64 casting, datetime-to-fractional-days
    conversion, and column renaming.

    Args:
        df: Polars DataFrame with columns: indicator, value, anchor_time.

    Returns:
        Wide-format Polars DataFrame with 'time' column and one column per indicator.
        Returns empty DataFrame if input is empty.
    """
    if df.is_empty():
        return pl.DataFrame()

    time_col = "anchor_time"
    if time_col not in df.columns:
        raise ValueError("Observation data must include an 'anchor_time' column.")

    # Parse string timestamps to datetime before pivoting so the
    # datetime→fractional-days conversion below always triggers.
    if df.schema.get(time_col) == pl.Utf8:
        df = df.with_columns(
            pl.col(time_col).str.to_datetime(strict=False, time_zone="UTC").alias(time_col)
        )

    wide_data = (
        df.with_columns(pl.col("value").cast(pl.Float64, strict=False))
        .pivot(on="indicator", index=time_col, values="value", aggregate_function="mean")
        .sort(time_col)
    )

    if wide_data.schema[time_col] in (pl.Datetime, pl.Date):
        t0 = wide_data[time_col].min()
        wide_data = wide_data.with_columns(
            ((pl.col(time_col) - t0).dt.total_seconds() / SECONDS_PER_DAY).alias(time_col)
        )

    if time_col in wide_data.columns:
        wide_data = wide_data.rename({time_col: "time"})

    # --- Sparsity validation ---
    indicator_cols = [c for c in wide_data.columns if c != "time"]
    if indicator_cols:
        n_rows = wide_data.height
        per_indicator: list[str] = []
        total_null = 0
        total_cells = 0
        for col in indicator_cols:
            n_null = wide_data[col].null_count()
            n_obs = n_rows - n_null
            total_null += n_null
            total_cells += n_rows
            if n_null > 0:
                pct = n_null / n_rows * 100
                per_indicator.append(f"{col}: {n_obs}/{n_rows} observed ({pct:.0f}% missing)")

        if total_cells > 0:
            overall_pct = total_null / total_cells * 100
            if overall_pct > 50:
                logger.warning(
                    "Sparse observation matrix: %.0f%% missing (%d/%d cells). "
                    "Multi-granularity indicators may cause excessive sparsity. "
                    "Per-indicator: %s",
                    overall_pct,
                    total_null,
                    total_cells,
                    "; ".join(per_indicator) if per_indicator else "all complete",
                )
            elif per_indicator:
                logger.info(
                    "Observation matrix sparsity: %.0f%% missing. %s",
                    overall_pct,
                    "; ".join(per_indicator),
                )

    return wide_data
