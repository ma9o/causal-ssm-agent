"""Capability-evaluation registry: ``(scenario x target)`` cells across the pipeline.

See :mod:`~evaluation.contracts` for the Scenario / TargetRunner /
TargetScorer protocols and :mod:`~evaluation.registry` for the
entry catalog. Import :mod:`~evaluation.seeds` to populate the
default rows (the Target 2 identification gates).
"""

from __future__ import annotations

from .contracts import (
    Cadence,
    Cost,
    Mode,
    Scenario,
    Score,
    Target,
    TargetRunner,
    TargetScorer,
)
from .registry import REGISTRY, RegistryEntry, evaluate, register, select

__all__ = [
    "REGISTRY",
    "Cadence",
    "Cost",
    "Mode",
    "RegistryEntry",
    "Scenario",
    "Score",
    "Target",
    "TargetRunner",
    "TargetScorer",
    "evaluate",
    "register",
    "select",
]
