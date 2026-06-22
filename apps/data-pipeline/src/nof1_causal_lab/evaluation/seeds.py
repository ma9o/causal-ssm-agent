"""Default registry seed: the zero-refactor Stage 2 identification gates.

Importing this module registers the seed entries into the global ``REGISTRY``.
The Stage 4 recovery row is intentionally not auto-registered yet: its runner
requires lifting ``benchmark._parameter_recovery`` so posterior sample names
align with the ``synthetic_nonlinear`` ``TRUE_*`` truth (see
:mod:`~nof1_causal_lab.evaluation.scorers.recovery`).
"""

from __future__ import annotations

from .contracts import Cadence, Cost, Mode, Stage
from .registry import RegistryEntry, register
from .scenarios import identification as id_scenarios
from .scenarios.recovery import RecoveryScenario
from .scorers.identification import IdentificationRunner, IdentificationScorer
from .scorers.recovery import RecoveryRunner, RecoveryScorer

_ID_RUNNER = IdentificationRunner()
_ID_SCORER = IdentificationScorer()


def _id_entry(scenario: id_scenarios.IdentificationScenario) -> RegistryEntry:
    return RegistryEntry(
        scenario=scenario,
        stage=Stage.IDENTIFICATION,
        runner=_ID_RUNNER,
        scorer=_ID_SCORER,
        cost=Cost.FREE,
        mode=Mode.GATE,
        cadence=Cadence.CI,
    )


# Recovery is a graded benchmark, not a gate. The smoke scenario carries
# precomputed draws so the benchmark path is exercised end-to-end; real
# synthetic_nonlinear draws (via lifting ``benchmark._parameter_recovery`` into
# a runner) replace it once that extraction lands.
RECOVERY_SMOKE = RecoveryScenario(
    name="recovery_smoke",
    capability="recovery:coverage_smoke",
    true_params={"theta": 0.5},
    model_inputs={"samples": {"theta": [i / 1000.0 for i in range(1001)]}},
)

_RECOVERY_ENTRY = RegistryEntry(
    scenario=RECOVERY_SMOKE,
    stage=Stage.INFERENCE,
    runner=RecoveryRunner(),
    scorer=RecoveryScorer(),
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
