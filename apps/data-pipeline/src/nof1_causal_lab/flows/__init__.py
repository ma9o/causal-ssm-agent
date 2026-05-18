"""Shared utilities for Prefect flow modules."""

from __future__ import annotations

import logging
from typing import Any

from prefect import get_run_logger
from prefect.context import get_run_context
from prefect.exceptions import MissingContextError

PrefectLogger = logging.Logger | logging.LoggerAdapter[logging.Logger]


class _PrefectAwareLogger:
    """Use Prefect run logging when available, otherwise fall back to stdlib."""

    def __init__(self, fallback: logging.Logger) -> None:
        self._fallback = fallback

    def _active(self) -> PrefectLogger:
        try:
            return get_run_logger()
        except MissingContextError:
            return self._fallback

    def __getattr__(self, name: str) -> Any:
        return getattr(self._active(), name)


def get_prefect_logger(name: str) -> _PrefectAwareLogger:
    return _PrefectAwareLogger(logging.getLogger(name))


def get_current_flow_run_id() -> str:
    """Return the current Prefect flow run id with a concrete runtime contract."""
    context = get_run_context()
    flow_run = getattr(context, "flow_run", None)
    flow_run_id = getattr(flow_run, "id", None)
    if flow_run_id is None:
        raise RuntimeError("Current Prefect context does not expose a flow run id")
    return str(flow_run_id)


__all__ = ["get_current_flow_run_id", "get_prefect_logger"]
