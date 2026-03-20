from pathlib import Path

import polars as pl

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.utils.causal_spec import (
    get_effective_observation_window,
)
from causal_ssm_agent.utils.config import get_config  # also loads .env
from causal_ssm_agent.utils.observation_semantics import (
    AnchorPolicy,
    get_anchor_policy,
    get_summary_operator,
    get_support_kind,
)
from causal_ssm_agent.utils.storage import get_base_uri, join

logger = get_prefect_logger(__name__)

SECONDS_PER_DAY = 86400.0
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

# Local-only Path — used by eval scripts and PROCESSED_DIR.  NOT remote-aware.
DATA_DIR = Path(get_base_uri()) if not get_base_uri().startswith("s3://") else Path.cwd() / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def input_dir(user_id: str) -> str:
    """Return the input directory for a user ID: ``data/{user_id}/input/``."""
    return join(DATA_URI, user_id, "input")


def runs_dir(user_id: str) -> str:
    """Return the single run directory for a user ID: ``data/{user_id}/run/``."""
    return join(DATA_URI, user_id, "run")


def get_orchestrator_chunk_size() -> int:
    """Get chunk size for stage 1 orchestrator."""
    return get_config().stage1_structure_proposal.chunk_size


def get_worker_chunk_size() -> int:
    """Get chunk size for stage 2 workers."""
    return get_config().stage2_workers.chunk_size


def get_sample_chunks() -> int:
    """Get number of sample chunks for stage 1 structure proposal."""
    return get_config().stage1_structure_proposal.sample_chunks


# Module-level defaults (evaluated at import time from config)
CHUNK_SIZE = get_orchestrator_chunk_size()
SAMPLE_CHUNKS = get_sample_chunks()


def load_lines(path: Path) -> list[str]:
    """Load individual lines from a preprocessed file."""
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def chunk_lines(lines: list[str], chunk_size: int) -> list[str]:
    """Group lines into chunks joined by newlines.

    Args:
        lines: Individual text lines
        chunk_size: Lines per chunk

    Returns:
        List of chunks, each a newline-joined group of lines
    """
    chunks = []
    for i in range(0, len(lines), chunk_size):
        batch = lines[i : i + chunk_size]
        chunks.append("\n".join(batch))
    return chunks


def load_text_chunks(path: Path, chunk_size: int | None = None) -> list[str]:
    """Load text chunks from a preprocessed file.

    Each chunk is a group of contiguous lines joined by newlines.

    Args:
        path: Path to preprocessed file (one record per line)
        chunk_size: Lines per chunk (default: CHUNK_SIZE from config)

    Returns:
        List of chunks, where each chunk is multiple lines joined together
    """
    return chunk_lines(load_lines(path), chunk_size or CHUNK_SIZE)


def sample_chunks(
    input_file: Path,
    n: int,
    seed: int | None = None,
    chunk_size: int | None = None,
) -> list[str]:
    """Sample n chunks evenly spaced across the input file with jitter.

    Args:
        input_file: Path to preprocessed file
        n: Number of chunks to sample
        seed: Random seed for reproducibility
        chunk_size: Lines per chunk (default: from config)

    Returns:
        List of sampled chunks
    """
    import random

    chunks = load_text_chunks(input_file, chunk_size=chunk_size)

    if seed is not None:
        random.seed(seed)

    if n <= 0:
        return []

    n = min(n, len(chunks))

    if n >= len(chunks):
        return chunks

    # Evenly space the samples across the dataset
    # Add small random jitter within each segment to avoid predictable sampling
    segment_size = len(chunks) / n
    sampled = []
    for i in range(n):
        segment_start = int(i * segment_size)
        segment_end = int((i + 1) * segment_size)
        # Pick randomly within this segment
        idx = random.randint(segment_start, segment_end - 1)
        sampled.append(chunks[idx])

    return sampled


_TIME_COLUMN_NAMES = (
    "timestamp",
    "time",
    "date",
    "datetime",
    "created_at",
    "ts",
    "dt",
    "updated_at",
)


def detect_time_column(df: pl.DataFrame) -> str:
    """Detect the primary time/date column in a DataFrame.

    Strategy:
    1. Look for Datetime/Date-typed columns; if exactly one, use it.
    2. If multiple, prefer common time column names.
    3. If none have datetime type, look for common names regardless of type.

    Raises:
        ValueError: If no time column can be identified.
    """
    dt_cols = [c for c in df.columns if df.schema[c] in (pl.Datetime, pl.Date)]
    if len(dt_cols) == 1:
        return dt_cols[0]
    if dt_cols:
        for name in _TIME_COLUMN_NAMES:
            if name in dt_cols:
                return name
        return dt_cols[0]

    # Fallback: look for common names regardless of type
    for name in _TIME_COLUMN_NAMES:
        if name in df.columns:
            return name

    raise ValueError(f"Could not detect time column in DataFrame with columns: {df.columns}")


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
        List of (tick_id, events_df) sorted chronologically.
        tick_id is an ISO-format string of the tick start time.
    """
    # Raw timestamps may already carry a timezone suffix such as `Z`.
    # Parse them as UTC so bucketing matches the rest of stage 2.
    if df.schema[time_col] == pl.Utf8:
        df = df.with_columns(
            pl.col(time_col).str.to_datetime(strict=False, time_zone="UTC").alias(time_col)
        )

    # Truncate to tick boundaries
    bucketed = df.with_columns(pl.col(time_col).dt.truncate(model_clock).alias("__tick__")).sort(
        time_col
    )

    # Group by tick
    result = []
    for tick_val, group_df in bucketed.group_by("__tick__", maintain_order=True):
        tick_dt = tick_val[0]
        tick_id = tick_dt.isoformat() if hasattr(tick_dt, "isoformat") else str(tick_dt)
        events = group_df.drop("__tick__")
        result.append((tick_id, events))

    return result


def observation_row_schema() -> dict[str, pl.DataType]:
    """Schema for canonical long-format observation rows."""
    return dict(OBSERVATION_ROW_SCHEMA)


def annotate_observation_rows(
    raw_data: pl.DataFrame,
    causal_spec: dict,
    *,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Attach observation metadata to long-format Stage 2 rows.

    The input ``time_col`` is the support-window start emitted by the computed
    and semantic extraction paths. The canonical observation-row contract keeps:
    - ``anchor_time``: latent-grid attachment time for the observation
    - ``support_start`` / ``support_end``: realized support bounds

    Canonical support semantics are always derived from the measurement spec,
    not preserved from any caller-supplied row metadata.
    """
    if raw_data.is_empty():
        return pl.DataFrame(schema=observation_row_schema())

    df = raw_data
    for col_name, dtype in OBSERVATION_ROW_SCHEMA.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

    if "indicator" not in df.columns:
        return df

    model_clock = causal_spec.get("measurement", {}).get("model_clock")
    indicator_rows = [
        {
            "indicator": ind["name"],
            "support_kind_meta": get_support_kind(ind),
            "summary_operator_meta": get_summary_operator(ind),
            "anchor_policy_meta": get_anchor_policy(ind),
            "observation_window_meta": get_effective_observation_window(ind, model_clock),
        }
        for ind in causal_spec.get("measurement", {}).get("indicators", [])
        if ind.get("name")
    ]
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
        pl.col(time_col).str.to_datetime(strict=False, time_zone="UTC")
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


def get_latest_preprocessed_file(
    directory: Path | None = None,
    exclude: set[str] | None = None,
) -> Path | None:
    """
    Find the most recently modified .txt file in the processed directory.

    Args:
        directory: Directory to search (default: data/processed/)
        exclude: Set of filenames to exclude (e.g., script outputs)

    Returns:
        Path to latest file, or None if no files found
    """
    search_dir = directory or (DATA_DIR / "processed")
    exclude = exclude or set()
    txt_files = [f for f in search_dir.glob("*.txt") if f.name not in exclude]

    if not txt_files:
        return None

    # Sort by modification time, newest first
    return max(txt_files, key=lambda p: p.stat().st_mtime)


def pivot_to_wide(raw_data: pl.DataFrame) -> pl.DataFrame:
    """Pivot long-format raw data to wide-format Polars DataFrame.

    Handles time column detection, Float64 casting, datetime-to-fractional-days
    conversion, and column renaming.

    Args:
        raw_data: Polars DataFrame with columns: indicator, value, anchor_time.

    Returns:
        Wide-format Polars DataFrame with 'time' column and one column per indicator.
        Returns empty DataFrame if input is empty.
    """
    if raw_data.is_empty():
        return pl.DataFrame()

    time_col = "anchor_time"
    if time_col not in raw_data.columns:
        raise ValueError("Raw observation data must include an 'anchor_time' column.")

    # Parse string timestamps to datetime before pivoting so the
    # datetime→fractional-days conversion below always triggers.
    if raw_data.schema.get(time_col) == pl.Utf8:
        raw_data = raw_data.with_columns(
            pl.col(time_col).str.to_datetime(strict=False, time_zone="UTC").alias(time_col)
        )

    wide_data = (
        raw_data.with_columns(pl.col("value").cast(pl.Float64, strict=False))
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
