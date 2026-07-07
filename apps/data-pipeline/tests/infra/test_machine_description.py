"""The static machine description served at GET /api/machine stays in lockstep with the graph."""

from nof1_causal_lab.episode_api import machine_description
from nof1_causal_lab.machine.graph import ARTIFACT_GRAPH, ROOT_ARTIFACTS
from nof1_causal_lab.machine.hierarchy import ACTIONS, CONTEXTS


def test_every_transition_declares_a_creation_class():
    valid = {"deterministic", "batch_llm", "judgment"}
    assert all(spec.creation_class in valid for spec in ARTIFACT_GRAPH)


def test_machine_description_serves_graph_and_classes():
    description = machine_description()

    stages = {entry["stage_id"]: entry for entry in description["stages"]}
    assert set(stages) == {spec.stage_id for spec in ARTIFACT_GRAPH}
    for spec in ARTIFACT_GRAPH:
        entry = stages[spec.stage_id]
        assert entry["consumes"] == list(spec.consumes)
        assert entry["produces"] == list(spec.produces)
        assert entry["produces_optional"] == list(spec.produces_optional)
        assert entry["derives"] == list(spec.derives)
        assert entry["creation_class"] == spec.creation_class
        assert entry["writable"] == spec.writable
    assert description["topological_stage_order"][0] == "stage-0"
    assert "question" in description["artifact_ids"]
    assert {entry["action_id"] for entry in description["actions"]} == {
        action.action_id for action in ACTIONS
    }
    assert {entry["context_id"] for entry in description["contexts"]} == {
        context.context_id for context in CONTEXTS
    }
    assert {entry["artifact_id"] for entry in description["roots"]} == set(ROOT_ARTIFACTS)
