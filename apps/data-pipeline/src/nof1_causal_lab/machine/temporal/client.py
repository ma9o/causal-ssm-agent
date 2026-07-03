"""Temporal client wiring shared by the facade and the worker."""

from __future__ import annotations

import os

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

EPISODE_TASK_QUEUE = os.environ.get("TEMPORAL_TASK_QUEUE", "nof1-episodes")


def episode_workflow_id(workspace_id: str) -> str:
    """One entity workflow per workspace episode."""
    return f"episode-{workspace_id}"


async def connect_client() -> Client:
    return await Client.connect(
        os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"),
        namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"),
        data_converter=pydantic_data_converter,
    )
