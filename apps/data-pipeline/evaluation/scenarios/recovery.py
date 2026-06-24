"""Parameter-recovery scenarios (diagnostic, Stage 4 inference).

A recovery scenario carries the model inputs and the known true parameter
values. The runner fits the model and the scorer checks credible-interval
coverage of the truth.

This module ships the generic, data-only :class:`RecoveryScenario` (precomputed
draws) and :class:`SyntheticNonlinearRecoveryScenario`, the fit-backed benchmark
over the synthetic-nonlinear fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.contracts import Capability, Kind, Scenario, Stage


@dataclass(frozen=True)
class RecoveryScenario(Scenario):
    """A model + data fixture with known true parameter values."""

    name: str
    capability: Capability
    true_params: dict[str, float]
    model_inputs: dict[str, Any] = field(default_factory=dict)
    kind: Kind = Kind.DIAGNOSTIC

    def inputs(self) -> dict[str, Any]:
        return self.model_inputs

    def truth_for(self, stage: Stage) -> Any | None:
        if stage is Stage.INFERENCE:
            return self.true_params
        return None


@dataclass(frozen=True)
class SyntheticNonlinearRecoveryScenario(Scenario):
    """The synthetic-nonlinear fixture as a fit-backed recovery benchmark.

    Truth is the fixture's ``RECOVERY_TARGETS`` (applied by the scorer's
    ``parameter_recovery`` call), so ``truth_for`` returns ``None``.
    """

    name: str = "synthetic_nonlinear_recovery"
    capability: Capability = Capability.RECOVERY
    kind: Kind = Kind.DIAGNOSTIC
    T: int = 32
    seed: int = 71
    method: str = "marginal_particle_gibbs"
    num_warmup: int = 500
    num_samples: int = 500

    def inputs(self) -> dict[str, Any]:
        return {
            "T": self.T,
            "seed": self.seed,
            "method": self.method,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
        }

    def truth_for(self, stage: Stage) -> Any | None:  # noqa: ARG002
        return None
