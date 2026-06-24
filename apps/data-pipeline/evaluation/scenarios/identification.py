"""Canonical identification scenarios (diagnostic, Stage 2 gates).

Each isolates the identification capability against a textbook graph with a
known verdict, fed straight to ``check_identifiability``. Truth is the expected
``{treatment -> identifiable?}`` map. The bow-arc case is the differentiator:
the system must *refuse* (declare non-identifiable) rather than emit a number.

User-facing graphs use explicit latent confounders (an unobserved construct
``U``), never bidirected edges — matching the project's modeling convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.contracts import Capability, Kind, Scenario, Stage


def _construct(name: str, *, is_outcome: bool = False) -> dict[str, Any]:
    return {"name": name, "is_outcome": is_outcome, "temporal_status": "time_invariant"}


def _edge(cause: str, effect: str) -> dict[str, Any]:
    return {"cause": cause, "effect": effect, "lagged": False}


def _observe(*constructs: str) -> dict[str, Any]:
    return {"indicators": [{"name": f"{c.lower()}_obs", "construct_name": c} for c in constructs]}


@dataclass(frozen=True)
class IdentificationScenario(Scenario):
    """A graph + observation set with a known identification verdict."""

    name: str
    capability: Capability
    latent_model: dict[str, Any]
    measurement_model: dict[str, Any]
    expected: dict[str, bool]  # treatment -> identifiable?
    iv_allowed: bool = True
    kind: Kind = Kind.DIAGNOSTIC

    def inputs(self) -> dict[str, Any]:
        return {
            "latent_model": self.latent_model,
            "measurement_model": self.measurement_model,
            "iv_allowed": self.iv_allowed,
        }

    def truth_for(self, stage: Stage) -> Any | None:
        if stage is Stage.IDENTIFICATION:
            return self.expected
        return None


# X -> Y, both observed: backdoor-trivially identifiable.
DIRECT_EFFECT = IdentificationScenario(
    name="id_direct_effect",
    capability=Capability.IDENTIFICATION,
    latent_model={
        "constructs": [_construct("X"), _construct("Y", is_outcome=True)],
        "edges": [_edge("X", "Y")],
    },
    measurement_model=_observe("X", "Y"),
    expected={"X": True},
)

# Bow arc: X -> Y with an unobserved confounder U -> X, U -> Y and no
# instrument. The effect is NOT identifiable; the system must refuse.
BOW_ARC = IdentificationScenario(
    name="id_bow_arc",
    capability=Capability.IDENTIFICATION,
    latent_model={
        "constructs": [
            _construct("X"),
            _construct("Y", is_outcome=True),
            _construct("U"),
        ],
        "edges": [_edge("X", "Y"), _edge("U", "X"), _edge("U", "Y")],
    },
    measurement_model=_observe("X", "Y"),  # U unobserved
    expected={"X": False},
    iv_allowed=True,
)

# Front door: X -> M -> Y with unobserved U -> X, U -> Y; observe X, M, Y.
# Identifiable via the front-door criterion (do-calculus).
FRONT_DOOR = IdentificationScenario(
    name="id_front_door",
    capability=Capability.IDENTIFICATION,
    latent_model={
        "constructs": [
            _construct("X"),
            _construct("M"),
            _construct("Y", is_outcome=True),
            _construct("U"),
        ],
        "edges": [
            _edge("X", "M"),
            _edge("M", "Y"),
            _edge("U", "X"),
            _edge("U", "Y"),
        ],
    },
    measurement_model=_observe("X", "M", "Y"),
    expected={"X": True},
)

# Instrument: Z -> X -> Y with unobserved U -> X, U -> Y; observe X, Y, Z.
# Identifiable only via IV under the parametric linearity assumption.
INSTRUMENT = IdentificationScenario(
    name="id_instrument",
    capability=Capability.IDENTIFICATION,
    latent_model={
        "constructs": [
            _construct("X"),
            _construct("Y", is_outcome=True),
            _construct("Z"),
            _construct("U"),
        ],
        "edges": [
            _edge("Z", "X"),
            _edge("X", "Y"),
            _edge("U", "X"),
            _edge("U", "Y"),
        ],
    },
    measurement_model=_observe("X", "Y", "Z"),
    expected={"X": True},
    iv_allowed=True,
)

ALL: list[IdentificationScenario] = [DIRECT_EFFECT, BOW_ARC, FRONT_DOOR, INSTRUMENT]
