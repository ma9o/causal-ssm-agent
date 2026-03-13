"""Stage 0: Agentic data ingestion (Prefect wrapper).

Accepts the most recent uploaded file for a user. Zip archives are extracted;
all other files are staged into a temporary directory for the ingestion agent.
"""

import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile, is_zipfile

from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.data import input_dir
from causal_ssm_agent.utils.llm import LLMStageContext

from .. import get_prefect_logger
from .stage0_ingest import IngestionResult, run_agentic_ingestion

logger = get_prefect_logger(__name__)


def _find_raw_input(code: str) -> Path:
    """Find the raw input file for a session code.

    Searches data/{code}/input/ for the most recent uploaded file.
    """
    user_dir = input_dir(code)
    if not user_dir.is_dir():
        raise FileNotFoundError(f"No raw data directory: {user_dir}")

    files = sorted(
        (path for path in user_dir.iterdir() if path.is_file() and not path.name.startswith(".")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if files:
        return files[0]

    raise FileNotFoundError(f"No files in {user_dir}")


def _prepare_raw_input(raw_path: Path, dest_dir: Path) -> Path:
    """Prepare a raw input file in a directory for the ingestion agent.

    Zip archives are extracted into ``dest_dir``. All other files are copied
    into ``dest_dir`` unchanged so the agent can inspect them via ``DATA_DIR``.

    Returns:
        Path to the prepared input directory (``dest_dir``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if is_zipfile(raw_path):
        with ZipFile(raw_path, "r") as zf:
            zf.extractall(dest_dir)
        return dest_dir

    shutil.copy2(raw_path, dest_dir / raw_path.name)
    return dest_dir


@task(cache_policy=INPUTS, persist_result=True, result_serializer="pickle")
async def agentic_ingest(code: str = "test_user") -> IngestionResult:
    """Ingest raw data using an LLM agent.

    Finds the most recent file in data/{code}/input/, prepares it in a
    temporary directory, and runs the agentic ingestion loop to produce a
    Polars DataFrame.

    Args:
        code: Session code (directory under data/)

    Returns:
        IngestionResult with DataFrame, source label, and column descriptions.
    """
    raw_path = _find_raw_input(code)
    logger.info("Ingesting %s from %s/", raw_path.name, raw_path.parent.name)

    config = get_config()
    async with LLMStageContext("stage-0") as ctx:
        generate = ctx.make_generate(config.stage0_ingestion.model)

        with tempfile.TemporaryDirectory(prefix="ingest_") as tmpdir:
            extract_dir = _prepare_raw_input(raw_path, Path(tmpdir))
            result = await run_agentic_ingestion(extract_dir, generate)

        # Attach trace for web persistence
        trace_out = ctx.finalize({})
        if "llm_trace" in trace_out:
            result.llm_trace = trace_out["llm_trace"]

        logger.info(
            "Ingested %d rows x %d columns", result.dataframe.shape[0], result.dataframe.shape[1]
        )
        return result
