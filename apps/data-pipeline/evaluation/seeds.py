"""Default registry seed: Target 2 identification gates + the Target 4 recovery row.

Importing this module registers the seed entries into the global ``REGISTRY``:
the identification gates (FREE/CI) and the synthetic-nonlinear recovery benchmark
(COMPUTE/MANUAL), which fits the fixture and scores coverage of every target via
the lifted ``evaluation.recovery`` extraction.
"""

from __future__ import annotations

from .contracts import Cadence, Cost, Mode, Target
from .registry import RegistryEntry, register
from .scenarios import identification as id_scenarios
from .scenarios.recovery import SyntheticNonlinearRecoveryScenario
from .scorers.identification import IdentificationRunner, IdentificationScorer
from .scorers.recovery import SyntheticNonlinearRecoveryRunner, SyntheticNonlinearRecoveryScorer

_ID_RUNNER = IdentificationRunner()
_ID_SCORER = IdentificationScorer()


def _id_entry(scenario: id_scenarios.IdentificationScenario) -> RegistryEntry:
    return RegistryEntry(
        scenario=scenario,
        target=Target.IDENTIFICATION,
        runner=_ID_RUNNER,
        scorer=_ID_SCORER,
        cost=Cost.FREE,
        mode=Mode.GATE,
        cadence=Cadence.CI,
    )


# Recovery is a graded, on-demand benchmark: it fits the synthetic-nonlinear
# fixture and scores coverage of every target via the lifted parameter_recovery
# extraction. COMPUTE/MANUAL — not a CI gate.
_RECOVERY_ENTRY = RegistryEntry(
    scenario=SyntheticNonlinearRecoveryScenario(),
    target=Target.INFERENCE,
    runner=SyntheticNonlinearRecoveryRunner(),
    scorer=SyntheticNonlinearRecoveryScorer(),
    cost=Cost.COMPUTE,
    mode=Mode.BENCHMARK,
    cadence=Cadence.MANUAL,
)

SEED_ENTRIES: list[RegistryEntry] = [
    *(_id_entry(s) for s in id_scenarios.ALL),
    _RECOVERY_ENTRY,
]

for _entry in SEED_ENTRIES:
    register(_entry)
