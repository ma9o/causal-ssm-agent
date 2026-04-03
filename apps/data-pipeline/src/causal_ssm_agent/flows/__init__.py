"""Shared utilities for Prefect flow modules."""

from __future__ import annotations

import logging
from typing import Any

from prefect import get_run_logger
from prefect.exceptions import MissingContextError


class _PrefectAwareLogger:
    """Use Prefect run logging when available, otherwise fall back to stdlib."""

    def __init__(self, fallback: logging.Logger) -> None:
        self._fallback = fallback

    def _active(self) -> logging.Logger:
        try:
            return get_run_logger()
        except MissingContextError:
            return self._fallback

    def __getattr__(self, name: str) -> Any:
        return getattr(self._active(), name)


def get_prefect_logger(name: str) -> _PrefectAwareLogger:
    return _PrefectAwareLogger(logging.getLogger(name))


__all__ = ["get_prefect_logger"]
