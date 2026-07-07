"""Declarative action/context hierarchy semantics."""

from nof1_causal_lab.flows.stage_tools import STAGE_TOOLS
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    ROOT_ARTIFACTS,
    ROOTS,
    WRITABLE_ARTIFACTS,
    stage_spec,
)
from nof1_causal_lab.machine.hierarchy import (
    ACTIONS,
    CONTEXTS,
    action_spec,
    context_spec,
    describe_actions,
    describe_contexts,
    legal_action_ids,
    primary_stage_action,
)


def _version(artifact_id, version=1, derived_from=None, produced_by=None, provenance="computed"):
    return ArtifactVersionInfo(
        artifact_id=artifact_id,
        version=version,
        provenance=provenance,
        derived_from=derived_from or {},
        produced_by=produced_by,
        created_at="2026-07-03T00:00:00Z",
    )


def _state(*infos):
    return EpisodeState().with_versions(list(infos))


def test_context_tree_is_closed():
    context_ids = {context.context_id for context in CONTEXTS}

    assert "navigator" in context_ids
    assert "episode-machine" in context_ids
    for context in CONTEXTS:
        if context.parent_id is not None:
            assert context.parent_id in context_ids
        if context.stage_id is not None:
            stage_spec(context.stage_id)


def test_stage_actions_cover_graph_exactly_once():
    stage_ids = {spec.stage_id for spec in ARTIFACT_GRAPH}
    action_stage_ids = {
        action.move.stage_id
        for action in ACTIONS
        if action.move is not None and action.move.kind == "run"
    }

    assert action_stage_ids == stage_ids
    for stage_id in stage_ids:
        action = primary_stage_action(stage_id)
        spec = stage_spec(stage_id)
        assert action.consumes == spec.consumes
        assert action.produces == spec.produces
        assert action.produces_optional == spec.produces_optional
        assert action.derives == spec.derives


def test_every_transition_declares_a_creation_class():
    valid = {"deterministic", "batch_llm", "judgment"}
    for spec in ARTIFACT_GRAPH:
        assert spec.creation_class in valid


def test_lower_context_tools_are_declared_stage_tools():
    tool_names_by_stage = {
        stage_id: {tool.name for tool in tools} for stage_id, tools in STAGE_TOOLS.items()
    }

    for context in CONTEXTS:
        if not context.allowed_tools or context.stage_id is None:
            continue
        declared = tool_names_by_stage[context.stage_id]
        assert set(context.allowed_tools).issubset(declared)


def test_actions_reference_declared_contexts():
    context_ids = {context.context_id for context in CONTEXTS}

    for action in ACTIONS:
        assert action.context_id in context_ids
        if action.lower_context_id is not None:
            assert action.lower_context_id in context_ids


def test_action_legality_is_artifact_state_only():
    empty = set(legal_action_ids(EpisodeState()))
    assert "nav.state" in empty
    assert "episode.create" in empty
    assert "episode.ingest_data" in empty
    assert "specify.constructs" not in empty
    assert "specify.edit" not in empty
    assert "analyze.save" not in empty
    assert "fit.compile" not in empty

    with_question = set(legal_action_ids(_state(_version("question", provenance="human"))))
    assert "specify.constructs" in with_question
    assert "specify.model" not in with_question

    with_posterior = set(legal_action_ids(_state(_version("posterior", produced_by="stage-5b"))))
    assert "analyze.save" in with_posterior


def test_identification_gate_is_visible_at_action_level():
    state = _state(
        _version("question", provenance="human"),
        _version("raw_data", produced_by="stage-0"),
        _version("constructs", produced_by="stage-1a"),
        _version("causal_spec", produced_by="stage-1b"),
        _version("extraction_report", produced_by="stage-2"),
        _version("model_data", produced_by="stage-2"),
        _version("validation_report", produced_by="stage-3"),
    )

    legal = set(legal_action_ids(state))
    assert "analyze.validate" in legal
    assert "fit.compile" not in legal
    assert "analyze.rank" not in legal
    assert "analyze.simulate" not in legal

    identified = state.with_versions([_version("identification_report", produced_by="stage-1b")])
    legal_identified = set(legal_action_ids(identified))
    assert "fit.compile" in legal_identified


def test_writable_surface_is_roots_plus_writable_transitions():
    writable_produced = {spec.produces[0] for spec in ARTIFACT_GRAPH if spec.writable}
    assert set(WRITABLE_ARTIFACTS) == set(ROOT_ARTIFACTS) | writable_produced
    assert set(ROOT_ARTIFACTS).issubset(WRITABLE_ARTIFACTS)

    # identification_report is derived from causal_spec, never written directly.
    assert "identification_report" not in WRITABLE_ARTIFACTS
    assert stage_spec("stage-1b").derives == ("identification_report",)


def test_roots_declare_their_write_pins():
    roots = {root.artifact_id: root for root in ROOTS}
    # A saved scenario pins the posterior it was simulated against, so it stales
    # through the same provenance mechanism when the posterior moves.
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
    assert edit["derives"] == ["identification_report"]
    assert action_spec("fit.compile").lower_context_id == "stage-4.model-spec"
    assert context_spec("stage-4.model-spec").runtime_state
