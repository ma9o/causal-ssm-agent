"""Parameter-recovery coverage scorer (Stage 4 inference).

Given posterior draws per parameter and the known true values, scores the
fraction whose true value falls inside a central credible interval — the
graded, benchmark-mode counterpart to ``tests.ssm_test_utils.assert_recovery_ci``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from nof1_causal_lab.evaluation.contracts import (
    Mode,
    Scenario,
    Score,
    Stage,
    StageRunner,
    StageScorer,
)


class RecoveryRunner(StageRunner):
    """Supplies precomputed posterior draws for a recovery scenario.

    Recovery is a graded, on-demand (MANUAL) benchmark: posterior draws are
    produced out-of-band by ``fit`` (e.g. in ``scripts/benchmarks``) and dropped
    into the scenario as ``model_inputs['samples']``; this runner hands them to
    the scorer for coverage grading.
    """

    stage = Stage.INFERENCE

    def run(self, scenario: Scenario) -> dict[str, Any]:
        return scenario.inputs()["samples"]


def interval_covers(
    draws: Any,
    true_value: float,
    q_low: float = 5.0,
    q_high: float = 95.0,
) -> bool:
    """Whether ``true_value`` falls inside the ``[q_low, q_high]`` percentile band."""
    arr = jnp.asarray(draws)
    lo = float(jnp.percentile(arr, q_low))
    hi = float(jnp.percentile(arr, q_high))
    return lo <= true_value <= hi


class RecoveryScorer(StageScorer):
    """Scores credible-interval coverage of known true parameter values."""

    stage = Stage.INFERENCE

    def __init__(self, q_low: float = 5.0, q_high: float = 95.0) -> None:
        self.q_low = q_low
        self.q_high = q_high

    def score(self, produced: dict[str, Any], truth: dict[str, float]) -> Score:
        per_param: dict[str, dict[str, Any]] = {}
        n_covered = 0
        for name, true_value in truth.items():
            covered = interval_covers(produced[name], true_value, self.q_low, self.q_high)
            per_param[name] = {"true": true_value, "covered": covered}
            n_covered += int(covered)

        return Score(
            name="recovery_coverage",
            mode=Mode.BENCHMARK,
            value=n_covered / len(truth) if truth else 1.0,
            detail={
                "n_params": len(truth),
                "q_low": self.q_low,
                "q_high": self.q_high,
                "per_param": per_param,
            },
        )
