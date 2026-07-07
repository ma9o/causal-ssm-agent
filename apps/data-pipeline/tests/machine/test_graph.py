"""Artifact-graph consistency and topology."""

import pytest

from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    DERIVATIONS,
    Derivation,
    Transition,
    producer_of,
    topological_transition_order,
    transition_spec,
)


def test_every_artifact_has_at_most_one_creator():
    seen = {}
    for spec in ARTIFACT_GRAPH:
        for artifact in spec.all_produces:
            assert artifact not in seen, (
                f"{artifact} produced by {seen[artifact]} and {spec.transition_id}"
            )
            seen[artifact] = spec.transition_id
    for spec in DERIVATIONS:
        assert spec.produces not in seen, f"{spec.produces} has two creators"
        seen[spec.produces] = "derived"


def test_topological_order_respects_runnable_transition_dependencies():
    order = topological_transition_order()
    position = {artifact_id: idx for idx, artifact_id in enumerate(order)}
    assert order[0] == "raw_data"
    for spec in ARTIFACT_GRAPH:
        for artifact in spec.consumes:
            producer = producer_of(artifact)
            if isinstance(producer, Transition):
                assert position[producer.transition_id] < position[spec.transition_id]


def test_derivations_are_standalone_multi_parent_nodes():
    derivations = {spec.produces: spec for spec in DERIVATIONS}

    assert derivations["causal_design"].from_ == (
        "latent_structure",
        "measurement_structure",
    )
    assert derivations["identification_report"].from_ == ("causal_design",)
    assert derivations["identification_report"].optional
    assert derivations["validation_report"].from_ == ("panel", "causal_design")
    assert derivations["compiled_ssm"].from_ == ("statistical_model_spec", "causal_design")

    transition_outputs = {artifact for spec in ARTIFACT_GRAPH for artifact in spec.all_produces}
    assert not transition_outputs.intersection(derivations)
    assert all(isinstance(producer_of(spec.produces), Derivation) for spec in DERIVATIONS)


def test_epistemic_gate_is_structural():
    assert "identification_report" in transition_spec("statistical_model_spec").consumes
    assert "identification_report" in transition_spec("baseline_report").consumes
    assert "identification_report" not in transition_spec("measurement_structure").all_produces


def test_panel_gate_is_structural():
    assert transition_spec("measurements").produces_optional == ("panel",)
    assert "panel" in transition_spec("statistical_model_spec").consumes
    assert "panel" in transition_spec("posterior").consumes


def test_deleted_stage_nodes_are_not_runnable_transitions():
    for artifact_id in ("causal_design", "validation_report", "compiled_ssm"):
        with pytest.raises(KeyError, match="Unknown transition"):
            transition_spec(artifact_id)


def test_unknown_transition_raises():
    with pytest.raises(KeyError, match="Unknown transition"):
        transition_spec("saved_scenarios")
