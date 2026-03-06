"""Stage 0: Agentic data ingestion (Prefect wrapper).

Accepts any .zip archive, extracts it, and uses an LLM agent to parse
the contents into a single Polars DataFrame.
"""

import logging
import tempfile

from pathlib import Path
from zipfile import ZipFile, is_zipfile

from inspect_ai.model import get_model
from prefect import task

from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.data import RAW_DIR
from causal_ssm_agent.utils.llm import (
    attach_trace,
    make_generate_fn,
    make_live_trace_path,
)

from .stage0_ingest import IngestionResult, run_agentic_ingestion

logger = logging.getLogger(__name__)


def _find_raw_input(user_id: str) -> Path:
    """Find the raw input file for a user.

    Searches data/raw/<user_id>/ for uploadable files.
    """
    user_dir = RAW_DIR / user_id
    if not user_dir.is_dir():
        raise FileNotFoundError(f"No raw data directory: {user_dir}")

    for pattern in ("*.zip",):
        files = sorted(user_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0]

    raise FileNotFoundError(f"No .zip files in {user_dir}")


def _extract_zip(archive_path: Path, dest_dir: Path) -> Path:
    """Extract a zip archive to a destination directory.

    Returns:
        Path to the extraction root (dest_dir).
    """
    with archive_path.open("rb") as f:
        if not is_zipfile(f):
            raise ValueError(f"{archive_path} is not a valid zip archive")

    with ZipFile(archive_path, "r") as zf:
        zf.extractall(dest_dir)

    return dest_dir


@task(result_serializer="pickle")
async def agentic_ingest(user_id: str = "test_user") -> IngestionResult:
    """Ingest raw data using an LLM agent.

    Finds the most recent .zip in data/raw/<user_id>/, extracts it,
    and runs the agentic ingestion loop to produce a Polars DataFrame.

    Args:
        user_id: User subdirectory under data/raw/

    Returns:
        IngestionResult with DataFrame, source label, and column descriptions.
    """
    raw_path = _find_raw_input(user_id)
    logger.info("Ingesting %s from %s/", raw_path.name, raw_path.parent.name)

    config = get_config()
    model = get_model(config.stage0_ingestion.model)
    trace_capture: dict = {}
    generate = make_generate_fn(
        model,
        trace_capture=trace_capture,
        trace_path=make_live_trace_path("stage-0"),
    )

    with tempfile.TemporaryDirectory(prefix="ingest_") as tmpdir:
        extract_dir = _extract_zip(raw_path, Path(tmpdir))
        result = await run_agentic_ingestion(extract_dir, generate)

    # Attach trace for web persistence
    result_meta: dict = {}
    attach_trace(result_meta, trace_capture)
    if "llm_trace" in result_meta:
        result.llm_trace = result_meta["llm_trace"]

    logger.info("Ingested %d rows x %d columns", result.dataframe.shape[0], result.dataframe.shape[1])
    return result
