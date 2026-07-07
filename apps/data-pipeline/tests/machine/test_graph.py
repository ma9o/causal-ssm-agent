"""Artifact-graph consistency and topology."""

import pytest

from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    producer_of,
    stage_spec,
    topological_stage_order,
)


def test_every_artifact_has_at_most_one_producer():
    seen = {}
    for spec in ARTIFACT_GRAPH:
        for artifact in spec.all_produces:
            assert artifact not in seen, (
                f"{artifact} produced by {seen[artifact]} and {spec.stage_id}"
            )
            seen[artifact] = spec.stage_id


def test_topological_order_respects_artifact_dependencies():
    order = topological_stage_order()
    position = {stage_id: idx for idx, stage_id in enumerate(order)}
    for spec in ARTIFACT_GRAPH:
        for artifact in spec.consumes:
            producer = producer_of(artifact)
            if producer is not None:
                assert position[producer.stage_id] < position[spec.stage_id]


def test_epistemic_gate_is_structural():
    """identification_report derives from causal_spec and is required downstream."""
    assert "identification_report" in stage_spec("stage-1b").derives
    assert "identification_report" not in stage_spec("stage-1b").produces_optional
    assert "identification_report" in stage_spec("stage-4").consumes
    assert "identification_report" in stage_spec("stage-6").consumes


def test_stage6_does_not_consume_question():
    assert "question" not in stage_spec("stage-6").consumes


def test_model_data_gate_is_structural():
    assert "model_data" in stage_spec("stage-2").produces_optional
    for downstream in ("stage-3", "stage-4", "stage-5b"):
        assert "model_data" in stage_spec(downstream).consumes


def test_unknown_stage_raises():
    with pytest.raises(KeyError, match="Unknown stage"):
        stage_spec("stage-99")
