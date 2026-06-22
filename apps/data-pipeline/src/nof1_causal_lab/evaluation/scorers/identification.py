"""Runner + scorer for the identification gates (Stage 2).

The runner imports the live ``check_identifiability`` — the same function the
stage and the causal tests call — so the gate scores the production code path,
not a copy.
"""

from __future__ import annotations

from typing import Any

from nof1_causal_lab.evaluation.contracts import (
    Mode,
    Scenario,
    Score,
    Stage,
    StageRunner,
    StageScorer,
)


class IdentificationRunner(StageRunner):
    """Drives the live y0 identification check on a scenario's graph."""

    stage = Stage.IDENTIFICATION

    def run(self, scenario: Scenario) -> dict[str, Any]:
        from nof1_causal_lab.utils.identifiability import check_identifiability

        ins = scenario.inputs()
        return check_identifiability(
            ins["latent_model"],
            ins["measurement_model"],
            iv_allowed=ins.get("iv_allowed", True),
        )


class IdentificationScorer(StageScorer):
    """Compares the produced ID verdict against the expected per-treatment map."""

    stage = Stage.IDENTIFICATION

    def score(self, produced: dict[str, Any], truth: dict[str, bool]) -> Score:
        identifiable = set(produced.get("identifiable_treatments", {}))
        non_identifiable = set(produced.get("non_identifiable_treatments", {}))

        results: dict[str, dict[str, Any]] = {}
        n_correct = 0
        for treatment, expected_id in truth.items():
            ok = (treatment in identifiable) if expected_id else (treatment in non_identifiable)
            results[treatment] = {
                "expected_identifiable": expected_id,
                "in_identifiable": treatment in identifiable,
                "in_non_identifiable": treatment in non_identifiable,
                "ok": ok,
            }
            n_correct += int(ok)

        return Score(
            name="identification",
            mode=Mode.GATE,
            passed=n_correct == len(truth),
            value=n_correct / len(truth) if truth else 1.0,
            detail={"treatments": results},
        )
