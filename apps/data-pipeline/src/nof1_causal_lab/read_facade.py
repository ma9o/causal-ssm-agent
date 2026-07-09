"""Read-only episode facade: the hosted viewer's entire backend.

Serves the same journal-backed read endpoints as the full tool server —
same code, same projections — against whatever store the environment
selects (R2 in production), without importing the tool-execution/SSM
stack. Deployments set ``EPISODE_FACADE_READ_ONLY=1`` so the move plane
403s and ``/api/capabilities`` advertises ``moves_enabled: false``; no
Temporal, no tool execution, no LLM anywhere. A published workspace is
viewable (including live, while a local service is still writing to it)
without any hosted stateful service.

Deployed as a Modal ASGI app (see :mod:`nof1_causal_lab.flows.modal_runners`),
or run locally::

    EPISODE_FACADE_READ_ONLY=1 uv run uvicorn --factory \
        nof1_causal_lab.read_facade:create_read_facade_app --port 8100
"""

from __future__ import annotations

from fastapi import FastAPI

from nof1_causal_lab.episode_api import (
    capabilities_router,
    machine_router,
    uploads_router,
    workspaces_router,
)
from nof1_causal_lab.episode_api import router as episode_router


def create_read_facade_app() -> FastAPI:
    app = FastAPI(title="Episode Read Facade")
    app.include_router(episode_router)
    app.include_router(capabilities_router)
    app.include_router(workspaces_router)
    app.include_router(uploads_router)
    app.include_router(machine_router)
    return app
