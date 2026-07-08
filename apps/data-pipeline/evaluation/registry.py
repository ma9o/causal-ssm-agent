"""The capability-evaluation registry: ``(scenario x target)`` cells with tiers.

Each :class:`RegistryEntry` pairs a scenario+target with the runner that drives
the live core and the scorer that grades it, tagged with a cost tier, a mode
(gate vs benchmark) and a cadence. The content lives here once; the execution
surfaces (pytest / CLI / Inspect) select rows by those tags.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    Cadence,
    Capability,
    Cost,
    Kind,
    Mode,
    Scenario,
    Score,
    Target,
    TargetRunner,
    TargetScorer,
)


@dataclass(frozen=True)
class RegistryEntry:
    """One cell of the evaluation matrix."""

    scenario: Scenario
    target: Target
    runner: TargetRunner
    scorer: TargetScorer
    cost: Cost
    mode: Mode
    cadence: Cadence

    @property
    def id(self) -> str:
        return f"{self.scenario.name}:{self.target.value}"


REGISTRY: list[RegistryEntry] = []


def register(entry: RegistryEntry) -> RegistryEntry:
    """Add ``entry`` to the global registry (idempotent by ``entry.id``)."""
    if any(existing.id == entry.id for existing in REGISTRY):
        return entry
    REGISTRY.append(entry)
    return entry


def select(
    *,
    target: Target | None = None,
    mode: Mode | None = None,
    cost: Cost | None = None,
    cadence: Cadence | None = None,
    kind: Kind | None = None,
    capability: Capability | None = None,
) -> list[RegistryEntry]:
    """Return registry entries matching every provided filter.

    ``target``/``mode``/``cost``/``cadence`` filter on the entry; ``kind`` and
    ``capability`` filter on the entry's scenario.
    """
    out = list(REGISTRY)
    if target is not None:
        out = [e for e in out if e.target == target]
    if mode is not None:
        out = [e for e in out if e.mode == mode]
    if cost is not None:
        out = [e for e in out if e.cost == cost]
    if cadence is not None:
        out = [e for e in out if e.cadence == cadence]
    if kind is not None:
        out = [e for e in out if e.scenario.kind == kind]
    if capability is not None:
        out = [e for e in out if e.scenario.capability == capability]
    return out


def evaluate(entry: RegistryEntry) -> Score:
    """Drive one cell: run the live core, then score against ground truth.

    Paid LLM cells are refused here — the cost gate is enforced at the registry
    boundary, not left to convention. Drive those through the Inspect backend.
    """
    if entry.cost == Cost.LLM_PAID:
        raise RuntimeError(
            f"refusing to auto-run paid LLM cell {entry.id!r}; "
            "drive it explicitly through the Inspect backend"
        )
    produced = entry.runner.run(entry.scenario)
    truth = entry.scenario.truth_for(entry.target)
    return entry.scorer.score(produced, truth)
