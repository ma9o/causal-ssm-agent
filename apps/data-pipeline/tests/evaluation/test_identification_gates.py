"""Identification gates: the registry's extraction verdicts must match canonical truth.

The bow-arc case is the differentiator — it asserts the system *refuses*
(declares non-identifiable) under latent confounding, which is the behavior no
forecasting or estimation benchmark can score.
"""

from __future__ import annotations

import pytest
from evaluation.contracts import Capability, Kind, Stage
from evaluation.registry import evaluate, select

from evaluation import seeds

_ID_ENTRIES = select(stage=Stage.IDENTIFICATION)


def test_kind_and_capability_filters():
    assert seeds.SEED_ENTRIES  # importing seeds populated the registry
    # Every seeded scenario is diagnostic so far; no integrative scenarios yet.
    assert select(kind=Kind.INTEGRATIVE) == []
    assert select(kind=Kind.DIAGNOSTIC), "expected diagnostic scenarios"
    assert len(select(capability=Capability.IDENTIFICATION)) == 4
    assert len(select(capability=Capability.RECOVERY)) == 1


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
