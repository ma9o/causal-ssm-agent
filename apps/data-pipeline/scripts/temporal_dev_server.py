"""Run a local Temporal dev server on a fixed port for the integration stack.

Uses the Temporal-provided dev server binary (auto-downloaded and cached by
the temporalio SDK on first use), so no separate `temporal` CLI install is
required.

State is ephemeral by default — each start begins with a clean history. Pass
``--db-filename`` to persist the event history to a SQLite file instead: then
the dev server behaves like a real cluster and in-flight episode workflows
resume exactly where they left off across a restart (the durable-execution
guarantee), rather than being orphaned. The agentic integration stack sets
this so restarting Temporal to pick up code or serve the UI never resets a
running episode.

The Temporal Web UI is served on ``port + 1000`` (8233 for the default
7233) so episode workflows, activities, retries, and event histories are
observable while the stack runs.

Usage:
    uv run python scripts/temporal_dev_server.py [--port 7233] [--db-filename PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("temporal-dev-server")


async def main(port: int, db_filename: str | None) -> None:
    from temporalio.testing import WorkflowEnvironment

    ui_port = port + 1000
    extra_args: list[str] = []
    if db_filename:
        # The dev server errors out if the db's parent dir is missing; create
        # it so a first run (before anything else touches .local/) succeeds.
        Path(db_filename).parent.mkdir(parents=True, exist_ok=True)
        extra_args = ["--db-filename", db_filename]
    env = await WorkflowEnvironment.start_local(
        port=port, ui=True, ui_port=ui_port, dev_server_extra_args=extra_args
    )
    logger.info("Temporal dev server listening on localhost:%d (namespace: default)", port)
    logger.info("Temporal Web UI at http://localhost:%d", ui_port)
    if db_filename:
        logger.info("History persisted to %s — workflows resume across restarts", db_filename)
    else:
        logger.info("History is ephemeral — a restart starts from a clean database")
    try:
        await asyncio.Event().wait()
    finally:
        await env.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7233)
    parser.add_argument(
        "--db-filename",
        default=None,
        help="Persist event history to this SQLite file so workflows resume across "
        "restarts. Omit for ephemeral (clean-slate) state.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.port, args.db_filename))
