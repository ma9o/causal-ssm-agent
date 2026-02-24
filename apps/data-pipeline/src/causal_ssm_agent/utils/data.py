import logging
from pathlib import Path

import polars as pl

from causal_ssm_agent.utils.config import get_config  # also loads .env

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QUERIES_DIR = DATA_DIR / "queries"
TRAINING_DIR = DATA_DIR / "training"


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
    search_dir = directory or PROCESSED_DIR
    exclude = exclude or set()
    txt_files = [f for f in search_dir.glob("*.txt") if f.name not in exclude]

    if not txt_files:
        return None

    # Sort by modification time, newest first
    return max(txt_files, key=lambda p: p.stat().st_mtime)


def resolve_query_path(filename: str) -> Path:
    """
    Resolve query file path from filename.

    Args:
        filename: Filename (mnemonic name like 'smoking-cancer.txt') in test-queries dir.
                  Extension is optional - will try .txt and .md if not provided.

    Returns:
        Full path to the query file

    Raises:
        FileNotFoundError: If no matching file found
    """
    # Try exact match first
    path = QUERIES_DIR / filename
    if path.exists():
        return path

    # Try with extensions if not provided
    if not path.suffix:
        for ext in [".txt", ".md"]:
            path_with_ext = QUERIES_DIR / f"{filename}{ext}"
            if path_with_ext.exists():
                return path_with_ext

    raise FileNotFoundError(f"Query file not found: {filename} in {QUERIES_DIR}")


def load_query(filename: str) -> str:
    """
    Load query content from file.

    Args:
        filename: Filename in test-queries dir (e.g., 'smoking-cancer')

    Returns:
        Query text content
    """
    path = resolve_query_path(filename)
    return path.read_text().strip()


def list_queries() -> list[str]:
    """List available query files in test-queries directory."""
    return [p.name for p in QUERIES_DIR.glob("*") if p.is_file() and p.name != ".gitkeep"]


def pivot_to_wide(raw_data: pl.DataFrame) -> pl.DataFrame:
    """Pivot long-format raw data to wide-format Polars DataFrame.

    Handles time column detection, Float64 casting, datetime-to-fractional-days
    conversion, and column renaming.

    Args:
        raw_data: Polars DataFrame with columns: indicator, value, and either
            timestamp (raw) or time_bucket (aggregated).

    Returns:
        Wide-format Polars DataFrame with 'time' column and one column per indicator.
        Returns empty DataFrame if input is empty.
    """
    if raw_data.is_empty():
        return pl.DataFrame()

    time_col = "time_bucket" if "time_bucket" in raw_data.columns else "timestamp"

    # Parse string timestamps to datetime before pivoting so the
    # datetime→fractional-days conversion below always triggers.
    if raw_data.schema.get(time_col) == pl.Utf8:
        raw_data = raw_data.with_columns(
            pl.col(time_col).str.to_datetime(strict=False).alias(time_col)
        )

    wide_data = (
        raw_data.with_columns(pl.col("value").cast(pl.Float64, strict=False))
        .pivot(on="indicator", index=time_col, values="value", aggregate_function="mean")
        .sort(time_col)
    )

    if wide_data.schema[time_col] in (pl.Datetime, pl.Date):
        t0 = wide_data[time_col].min()
        wide_data = wide_data.with_columns(
            ((pl.col(time_col) - t0).dt.total_seconds() / 86400.0).alias(time_col)
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
