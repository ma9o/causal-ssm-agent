"""Unit tests for the generic parameter-recovery coverage scorer."""

from __future__ import annotations

import jax.numpy as jnp

from nof1_causal_lab.evaluation import seeds
from nof1_causal_lab.evaluation.contracts import Mode, Stage
from nof1_causal_lab.evaluation.registry import evaluate, select
from nof1_causal_lab.evaluation.scenarios.recovery import RecoveryScenario
from nof1_causal_lab.evaluation.scorers.recovery import RecoveryScorer, interval_covers


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
        capability="recovery:demo",
        true_params={"a": 0.5},
    )
    assert scenario.truth_for(Stage.INFERENCE) == {"a": 0.5}
    assert scenario.truth_for(Stage.IDENTIFICATION) is None


def test_recovery_row_registered_and_evaluates():
    names = {entry.scenario.name for entry in seeds.SEED_ENTRIES}
    assert "recovery_smoke" in names

    rows = select(stage=Stage.INFERENCE)
    assert rows, "recovery row not registered"

    score = evaluate(rows[0])
    assert score.mode is Mode.BENCHMARK
    assert score.value == 1.0
