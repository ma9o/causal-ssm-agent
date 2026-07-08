"""Parameter-recovery coverage scorer (Target 4 inference).

Given posterior draws per parameter and the known true values, scores the
fraction whose true value falls inside a central credible interval — the
graded, benchmark-mode counterpart to ``tests.ssm_test_utils.assert_recovery_ci``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from evaluation.contracts import (
    Mode,
    Scenario,
    Score,
    Target,
    TargetRunner,
    TargetScorer,
)


class SyntheticNonlinearRecoveryRunner(TargetRunner):
    """Simulates the synthetic-nonlinear fixture, fits it, returns the InferenceResult.

    A graded, on-demand (MANUAL / COMPUTE) benchmark: it runs the production
    ``fit`` on the same fixture the recovery benchmark uses.
    """

    target = Target.INFERENCE

    def run(self, scenario: Scenario) -> Any:
        from evaluation.fixtures.synthetic_nonlinear import (
            build_synthetic_nonlinear_model,
            simulate_synthetic_nonlinear_data,
        )
        from nof1_causal_lab.models.ssm.inference import fit

        cfg = scenario.inputs()
        data = simulate_synthetic_nonlinear_data(T=cfg["T"], seed=cfg["seed"])
        model = build_synthetic_nonlinear_model(data)
        return fit(
            model,
            data.observations,
            data.times,
            method=cfg["method"],
            num_warmup=cfg["num_warmup"],
            num_samples=cfg["num_samples"],
        )


class SyntheticNonlinearRecoveryScorer(TargetScorer):
    """Scores synthetic-nonlinear recovery via the lifted ``parameter_recovery``.

    ``produced`` is the fit ``InferenceResult``; the truth lives in the fixture's
    ``RECOVERY_TARGETS`` and is applied inside ``parameter_recovery``, so the
    scenario truth is ignored.
    """

    target = Target.INFERENCE

    def score(self, produced: Any, truth: Any) -> Score:  # noqa: ARG002
        from evaluation.recovery.extraction import parameter_recovery

        rec = parameter_recovery(produced, elapsed_seconds=1.0)
        return Score(
            name="synthetic_nonlinear_recovery",
            mode=Mode.BENCHMARK,
            value=rec["summary"]["coverage_90"],
            detail={
                "site_count": rec["site_count"],
                "summary": rec["summary"],
                "by_family": rec["by_family"],
            },
        )


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


class RecoveryScorer(TargetScorer):
    """Scores credible-interval coverage of known true parameter values."""

    target = Target.INFERENCE

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
