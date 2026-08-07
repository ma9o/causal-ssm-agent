"""Declarative action/context hierarchy semantics."""

from nof1_causal_lab.flows.context_tools import CONTEXT_TOOLS
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    DERIVATIONS,
    ROOT_ARTIFACTS,
    ROOTS,
    WRITABLE_ARTIFACTS,
    transition_spec,
)
from nof1_causal_lab.machine.hierarchy import (
    ACTIONS,
    ACTIONS_BY_ID,
    CONTEXTS,
    CONTEXTS_BY_ID,
    describe_actions,
    describe_contexts,
    primary_transition_action,
)


def test_context_tree_is_closed():
    context_ids = {context.context_id for context in CONTEXTS}

    assert "navigator" in context_ids
    assert "episode-machine" in context_ids
    for context in CONTEXTS:
        if context.parent_id is not None:
            assert context.parent_id in context_ids


def test_transition_actions_cover_graph_exactly_once():
    transition_ids = {spec.transition_id for spec in ARTIFACT_GRAPH}
    action_transition_ids = {
        action.move.artifact_id
        for action in ACTIONS
        if action.move is not None and action.move.kind == "run"
    }

    assert action_transition_ids == transition_ids
    for artifact_id in transition_ids:
        action = primary_transition_action(artifact_id)
        spec = transition_spec(artifact_id)
        assert action.consumes == spec.consumes
        assert action.produces == (spec.produces,)
        assert action.produces_optional == spec.produces_optional


def test_every_transition_declares_a_creation_class():
    valid = {"deterministic", "batch_llm", "judgment"}
    for spec in ARTIFACT_GRAPH:
        assert spec.creation_class in valid


def test_public_context_tools_are_allowed_by_their_context():
    tool_names_by_context = {
        context_id: {tool.name for tool in tools} for context_id, tools in CONTEXT_TOOLS.items()
    }

    for context in CONTEXTS:
        if not context.allowed_tools:
            continue
        declared = tool_names_by_context[context.context_id]
        assert declared.issubset(context.allowed_tools)


def test_actions_reference_declared_contexts():
    context_ids = {context.context_id for context in CONTEXTS}

    for action in ACTIONS:
        assert action.context_id in context_ids
        if action.lower_context_id is not None:
            assert action.lower_context_id in context_ids


def test_writable_surface_is_roots_plus_writable_transitions():
    writable_produced = {spec.produces for spec in ARTIFACT_GRAPH if spec.writable}
    assert set(WRITABLE_ARTIFACTS) == set(ROOT_ARTIFACTS) | writable_produced
    assert set(ROOT_ARTIFACTS).issubset(WRITABLE_ARTIFACTS)

    derived_ids = {spec.produces for spec in DERIVATIONS}
    assert not derived_ids.intersection(WRITABLE_ARTIFACTS)
    assert "causal_design" in derived_ids
    assert "identification_report" in derived_ids


def test_roots_declare_their_write_pins():
    roots = {root.artifact_id: root for root in ROOTS}
    assert roots["saved_scenarios"].write_pins == ("posterior",)
    assert roots["question"].write_pins == ()


def test_registry_descriptions_are_json_ready():
    action_payload = describe_actions()
    context_payload = describe_contexts()

    assert {entry["action_id"] for entry in action_payload} == {
        action.action_id for action in ACTIONS
    }
    assert {entry["context_id"] for entry in context_payload} == {
        context.context_id for context in CONTEXTS
    }
    edit = next(entry for entry in action_payload if entry["action_id"] == "specify.edit")
    assert edit["derives"] == [
        "causal_design",
        "structural_plan",
        "identification_report",
        "validation_report",
        "compiled_ssm",
    ]
    assert ACTIONS_BY_ID["fit.specify"].lower_context_id == "statistical-model-spec"
    assert CONTEXTS_BY_ID["statistical-model-spec"].runtime_state
