"""Tests for the recovery scorers and the lifted parameter_recovery extraction."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from evaluation.contracts import Cadence, Capability, Cost, Mode, Stage
from evaluation.fixtures.synthetic_nonlinear import RECOVERY_TARGETS
from evaluation.registry import select
from evaluation.scenarios.recovery import RecoveryScenario
from evaluation.scorers.recovery import (
    RecoveryScorer,
    SyntheticNonlinearRecoveryScorer,
    interval_covers,
)

from evaluation import seeds


def test_interval_covers():
    draws = jnp.linspace(0.0, 1.0, 1001)
    assert interval_covers(draws, 0.5)
    assert not interval_covers(draws, 0.99)


def test_recovery_scorer_reports_per_param_coverage():
    produced = {"a": jnp.linspace(0.0, 1.0, 1001), "b": jnp.linspace(0.0, 1.0, 1001)}
    truth = {"a": 0.5, "b": 0.99}

    score = RecoveryScorer().score(produced, truth)

    assert score.mode is Mode.BENCHMARK
    assert score.value == 0.5
    assert score.detail["per_param"]["a"]["covered"] is True
    assert score.detail["per_param"]["b"]["covered"] is False


def test_recovery_scenario_truth_for():
    scenario = RecoveryScenario(
        name="rec_demo",
        capability=Capability.RECOVERY,
        true_params={"a": 0.5},
    )
    assert scenario.truth_for(Stage.INFERENCE) == {"a": 0.5}
    assert scenario.truth_for(Stage.IDENTIFICATION) is None


def test_synthetic_recovery_row_registered():
    """The fit-backed recovery row is registered as a COMPUTE/MANUAL benchmark.

    It is NOT evaluated here — that runs the production ``fit`` and belongs in the
    slow suite — only that it is wired into the registry with the right tier tags.
    """
    assert seeds.SEED_ENTRIES  # importing seeds populated the registry
    rows = [
        r
        for r in select(stage=Stage.INFERENCE)
        if r.scenario.name == "synthetic_nonlinear_recovery"
    ]
    assert rows, "synthetic recovery row not registered"
    row = rows[0]
    assert (row.cost, row.mode, row.cadence) == (Cost.COMPUTE, Mode.BENCHMARK, Cadence.MANUAL)


def _all_true_grouped_samples(chains: int = 2, draws: int = 16) -> dict[str, np.ndarray]:
    """Build ``{site: ndarray}`` where every posterior draw equals the target truth."""
    locs = []
    for label, target in RECOVERY_TARGETS.items():
        if isinstance(target, dict):
            raw = target["index"]
            index = tuple(int(i) for i in raw) if isinstance(raw, (list, tuple)) else (int(raw),)
            locs.append((str(target["site"]), index, float(target["true"])))
        else:
            locs.append((label, (), float(target)))

    shape_by_site: dict[str, tuple[int, ...]] = {}
    for site, index, _ in locs:
        cur = shape_by_site.get(site, ())
        need = tuple(i + 1 for i in index)
        n = max(len(cur), len(need))
        shape_by_site[site] = tuple(
            max(cur[k] if k < len(cur) else 0, need[k] if k < len(need) else 0) for k in range(n)
        )

    grouped = {
        site: np.zeros((chains, draws, *shape), dtype=np.float64)
        for site, shape in shape_by_site.items()
    }
    for site, index, true in locs:
        grouped[site][(..., *index)] = true
    return grouped


class _FakeMCMC:
    def __init__(self, grouped: dict[str, np.ndarray]) -> None:
        self._grouped = grouped

    def get_samples(self, group_by_chain: bool = False) -> dict[str, np.ndarray]:
        return self._grouped


class _FakeResult:
    def __init__(self, grouped: dict[str, np.ndarray]) -> None:
        self.diagnostics = {"mcmc": _FakeMCMC(grouped)}


def test_synthetic_recovery_scorer_full_coverage_on_exact_draws():
    """Every draw equal to truth => q05==q95==truth => 100% interval coverage.

    Exercises the lifted ``parameter_recovery`` end-to-end through the scorer
    without running ``fit`` (ssm-independent).
    """
    grouped = _all_true_grouped_samples()
    score = SyntheticNonlinearRecoveryScorer().score(_FakeResult(grouped), None)

    assert score.mode is Mode.BENCHMARK
    assert score.value == 1.0
    assert score.detail["site_count"] == len(RECOVERY_TARGETS)
