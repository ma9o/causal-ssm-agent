"""Identification gates: the registry's Stage 2 verdicts must match canonical truth.

The bow-arc case is the differentiator — it asserts the system *refuses*
(declares non-identifiable) under latent confounding, which is the behavior no
forecasting or estimation benchmark can score.
"""

from __future__ import annotations

import pytest

from nof1_causal_lab.evaluation import seeds
from nof1_causal_lab.evaluation.contracts import Stage
from nof1_causal_lab.evaluation.registry import evaluate, select

_ID_ENTRIES = select(stage=Stage.IDENTIFICATION)


def test_identification_gates_registered():
    names = {entry.scenario.name for entry in seeds.SEED_ENTRIES}
    assert names >= {
        "id_direct_effect",
        "id_bow_arc",
        "id_front_door",
        "id_instrument",
    }


@pytest.mark.parametrize("entry", _ID_ENTRIES, ids=[e.id for e in _ID_ENTRIES])
def test_identification_gate(entry):
    score = evaluate(entry)
    assert score.passed, f"{entry.id} failed: {score.detail}"
