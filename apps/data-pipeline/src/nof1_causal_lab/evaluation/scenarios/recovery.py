"""Parameter-recovery scenarios (diagnostic, Stage 4 inference).

A recovery scenario carries the model inputs and the known true parameter
values. The runner fits the model and the scorer checks credible-interval
coverage of the truth.

This module ships the generic, data-only :class:`RecoveryScenario`. Wiring the
``synthetic_nonlinear`` fixture to it requires lifting
``benchmark._parameter_recovery`` so posterior sample names align with the
fixture's ``TRUE_*`` constants; that runner is the immediate follow-up and is
deliberately not registered yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nof1_causal_lab.evaluation.contracts import Scenario, Stage


@dataclass(frozen=True)
class RecoveryScenario(Scenario):
    """A model + data fixture with known true parameter values."""

    name: str
    capability: str
    true_params: dict[str, float]
    model_inputs: dict[str, Any] = field(default_factory=dict)

    def inputs(self) -> dict[str, Any]:
        return self.model_inputs

    def truth_for(self, stage: Stage) -> Any | None:
        if stage is Stage.INFERENCE:
            return self.true_params
        return None
