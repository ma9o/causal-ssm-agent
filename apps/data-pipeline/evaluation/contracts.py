"""Core contracts for the capability-evaluation registry.

The pipeline is validated as a matrix of ``(scenario x stage)`` cells. A
:class:`Scenario` supplies inputs and ground truth for whichever stages it
covers; a :class:`StageRunner` drives the live stage core to produce an output;
a :class:`StageScorer` compares that output against the scenario's ground truth.

This module is backend-agnostic on purpose: the same registry entries are
driven by pytest (gates), CLI/Modal benchmark scripts (graded benchmarks), and
Inspect tasks (paid LLM cells). The Prefect flow, the test suite, and these
runners are all adapters over the same importable stage cores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Stage(StrEnum):
    """A pipeline stage a scenario can carry ground truth for.

    The enum grows with the matrix: a stage gains a member when its first
    scorer lands. Planned next: constructs (stage1a), measurement (stage1b),
    statistical_model_spec (stage4), effects (stage6).
    """

    IDENTIFICATION = "identification"  # stage1b: y0 identification verdict
    INFERENCE = "inference"  # stage5b: parameter recovery


class Cost(StrEnum):
    """Execution cost tier. ``LLM_PAID`` cells are never auto-run."""

    FREE = "free"  # pure, deterministic, CI-cheap
    COMPUTE = "compute"  # CPU sampling
    GPU = "gpu"  # GPU sampling (Modal)
    LLM_PAID = "llm_paid"  # spends model tokens


class Mode(StrEnum):
    """Whether a cell is a blocking correctness gate or a graded benchmark."""

    GATE = "gate"  # binary, CI-blocking
    BENCHMARK = "benchmark"  # graded, tracked over time, non-blocking


class Cadence(StrEnum):
    """How often a cell is expected to run.

    A ``NIGHTLY`` tier is added once a scheduled job consumes it.
    """

    CI = "ci"
    MANUAL = "manual"


class Kind(StrEnum):
    """Scenario breadth.

    ``DIAGNOSTIC`` scenarios isolate one capability so a failure is localizable;
    ``INTEGRATIVE`` scenarios (e.g. an end-to-end epidemic benchmark) exercise
    several capabilities together to test that they compose.
    """

    DIAGNOSTIC = "diagnostic"
    INTEGRATIVE = "integrative"


class Capability(StrEnum):
    """Controlled vocabulary for the capability a scenario exercises.

    Coarse on purpose — leaf detail (front-door vs bow-arc) lives in the scenario
    ``name``. Members are added as scenarios for new capabilities land.
    """

    IDENTIFICATION = "identification"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class Score:
    """Outcome of scoring one ``(scenario, stage)`` cell."""

    name: str
    mode: Mode
    passed: bool | None = None  # set for gates
    value: float | None = None  # set for benchmarks (and gates, as a fraction)
    detail: dict[str, Any] = field(default_factory=dict)


class Scenario(ABC):
    """A test case carrying inputs and ground truth for one or more stages."""

    name: str
    capability: Capability
    kind: Kind

    @abstractmethod
    def inputs(self) -> dict[str, Any]:
        """Inputs fed to the stage runner."""

    @abstractmethod
    def truth_for(self, stage: Stage) -> Any | None:
        """Ground truth for ``stage``, or ``None`` when this scenario omits it."""


class StageRunner(ABC):
    """Adapter that drives one stage's live core to produce an output."""

    stage: Stage

    @abstractmethod
    def run(self, scenario: Scenario) -> Any:
        """Invoke the live stage core on the scenario's inputs."""


class StageScorer(ABC):
    """Compares a stage's produced output against scenario ground truth."""

    stage: Stage

    @abstractmethod
    def score(self, produced: Any, truth: Any) -> Score:
        """Score ``produced`` against ``truth``."""
