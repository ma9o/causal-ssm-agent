"""The static machine description served at GET /api/machine stays in lockstep with the graph."""

from nof1_causal_lab.episode_api import machine_description
from nof1_causal_lab.machine.graph import ARTIFACT_GRAPH, DERIVATIONS, ROOT_ARTIFACTS
from nof1_causal_lab.machine.hierarchy import ACTIONS, CONTEXTS


def test_every_transition_declares_a_creation_class():
    valid = {"deterministic", "batch_llm", "judgment"}
    assert all(spec.creation_class in valid for spec in ARTIFACT_GRAPH)


def test_machine_description_serves_graph_and_classes():
    description = machine_description()

    transitions = {entry["transition_id"]: entry for entry in description["transitions"]}
    assert set(transitions) == {spec.transition_id for spec in ARTIFACT_GRAPH}
    for spec in ARTIFACT_GRAPH:
        entry = transitions[spec.transition_id]
        assert entry["consumes"] == list(spec.consumes)
        assert entry["produces"] == [spec.produces]
        assert entry["produces_optional"] == list(spec.produces_optional)
        assert entry["creation_class"] == spec.creation_class
        assert entry["writable"] == spec.writable
    derivations = {entry["produces"]: entry for entry in description["derivations"]}
    assert set(derivations) == {spec.produces for spec in DERIVATIONS}
    assert "validation_report" in description["topological_artifact_order"]
    assert description["topological_artifact_order"].index("measurements") < description[
        "topological_artifact_order"
    ].index("validation_report")
    assert description["topological_transition_order"][0] == "raw_data"
    assert "question" in description["artifact_ids"]
    assert {entry["action_id"] for entry in description["actions"]} == {
        action.action_id for action in ACTIONS
    }
    assert {entry["context_id"] for entry in description["contexts"]} == {
        context.context_id for context in CONTEXTS
    }
    assert {entry["artifact_id"] for entry in description["roots"]} == set(ROOT_ARTIFACTS)
