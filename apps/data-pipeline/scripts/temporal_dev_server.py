"""Run a local Temporal dev server on a fixed port for the integration stack.

Uses the Temporal-provided dev server binary (auto-downloaded and cached by
the temporalio SDK on first use), so no separate `temporal` CLI install is
required. State is ephemeral — each stack start begins with a clean history,
matching the old stack's fresh-database semantics.

Usage:
    uv run python scripts/temporal_dev_server.py [--port 7233]
"""

from __future__ import annotations

import argparse
import asyncio
import logging

logger = logging.getLogger("temporal-dev-server")


async def main(port: int) -> None:
    from temporalio.testing import WorkflowEnvironment

    env = await WorkflowEnvironment.start_local(port=port, ui=False)
    logger.info("Temporal dev server listening on localhost:%d (namespace: default)", port)
    try:
        await asyncio.Event().wait()
    finally:
        await env.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7233)
    args = parser.parse_args()
    asyncio.run(main(args.port))
