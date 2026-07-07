"""Declarative action/context hierarchy semantics."""

from nof1_causal_lab.flows.stage_tools import STAGE_TOOLS
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
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
    CONTEXTS,
    action_spec,
    context_spec,
    describe_actions,
    describe_contexts,
    legal_action_ids,
    primary_transition_action,
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
    runner_ids = {spec.runner_id for spec in ARTIFACT_GRAPH}

    assert "navigator" in context_ids
    assert "episode-machine" in context_ids
    for context in CONTEXTS:
        if context.parent_id is not None:
            assert context.parent_id in context_ids
        if context.runner_id is not None:
            assert context.runner_id in runner_ids


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


def test_lower_context_tools_are_declared_stage_tools():
    tool_names_by_stage = {
        stage_id: {tool.name for tool in tools} for stage_id, tools in STAGE_TOOLS.items()
    }

    for context in CONTEXTS:
        if not context.allowed_tools or context.runner_id is None:
            continue
        declared = tool_names_by_stage[context.runner_id]
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
    assert "specify.latent_structure" not in empty
    assert "specify.edit" not in empty
    assert "analyze.save" not in empty
    assert "fit.specify" not in empty

    with_question = set(legal_action_ids(_state(_version("question", provenance="human"))))
    assert "specify.latent_structure" in with_question
    assert "specify.measurement" not in with_question

    with_posterior = set(_state(_version("posterior", produced_by="stage-5b")).current)
    assert with_posterior == {"posterior"}
    legal_with_posterior = set(
        legal_action_ids(_state(_version("posterior", produced_by="stage-5b")))
    )
    assert "analyze.save" in legal_with_posterior


def test_identification_gate_is_visible_at_action_level():
    state = _state(
        _version("question", provenance="human"),
        _version("raw_data", produced_by="stage-0"),
        _version("latent_structure", produced_by="stage-1a"),
        _version("measurement_structure", produced_by="stage-1b"),
        _version("causal_design", produced_by="derive:causal_design"),
        _version("measurements", produced_by="stage-2"),
        _version("panel", produced_by="stage-2"),
        _version("validation_report", produced_by="derive:validation_report"),
    )

    legal = set(legal_action_ids(state))
    assert "measure.extract" in legal
    assert "fit.specify" not in legal
    assert "analyze.rank" not in legal
    assert "analyze.simulate" not in legal

    identified = state.with_versions(
        [_version("identification_report", produced_by="derive:identification_report")]
    )
    legal_identified = set(legal_action_ids(identified))
    assert "fit.specify" in legal_identified


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
        "identification_report",
        "validation_report",
        "compiled_ssm",
    ]
    assert action_spec("fit.specify").lower_context_id == "statistical-model-spec"
    assert context_spec("statistical-model-spec").runtime_state
