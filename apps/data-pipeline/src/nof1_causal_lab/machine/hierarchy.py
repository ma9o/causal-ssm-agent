"""Public naming and context layer above the artifact machine.

This module is intentionally thin and pure: no storage, no runners, no LLM
clients, no Modal, no web. It maps intent-named public actions onto machine
moves (``run``/``write``), reads, and derived queries, and it describes the
control/context hierarchy the harness navigates.

The *semantics* of creation — what a transition consumes, produces, derives,
its creation class, and whether it is writable — live in the machine core
(:mod:`nof1_causal_lab.machine.graph`). This layer only names and routes.

Machine legality (may a move happen?) is existence-only and lives in
:mod:`nof1_causal_lab.machine.moves`. The ``legal_actions`` here compute the
weaker, stricter-when-useful *affordance* set: which public actions are worth
surfacing for the navigator (e.g. do not offer ``analyze.save`` before a
``posterior`` exists, even though ``write(saved_scenarios)`` is always legal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    ROOT_ARTIFACTS,
    WRITABLE_ARTIFACTS,
    stage_spec,
)

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import ArtifactId, EpisodeState


ContextLayer = Literal["navigator", "registry", "machine", "delegated", "tool"]
ActionKind = Literal["read", "produce", "check", "query", "driver", "external"]
ActionMode = Literal["direct", "delegated", "async", "read"]
MoveKind = Literal["run", "write"]


@dataclass(frozen=True)
class ContextSpec:
    """One authority boundary in the harness/control hierarchy."""

    context_id: str
    layer: ContextLayer
    label: str
    parent_id: str | None = None
    stage_id: str | None = None
    owns: tuple[ArtifactId, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    runtime_state: tuple[str, ...] = ()


@dataclass(frozen=True)
class MachineMoveSpec:
    """The machine move an action compiles to, when it mutates state."""

    kind: MoveKind
    stage_id: str | None = None
    artifact_id: ArtifactId | None = None


@dataclass(frozen=True)
class ToolQuerySpec:
    """A read-only stage tool invoked from the query plane."""

    stage_id: str
    tool_name: str
    freshness_checked: bool = True


@dataclass(frozen=True)
class ActionSpec:
    """One public action exposed over MCP/RPC/SDK."""

    action_id: str
    namespace: str
    name: str
    kind: ActionKind
    mode: ActionMode
    context_id: str
    consumes: tuple[ArtifactId, ...] = ()
    produces: tuple[ArtifactId, ...] = ()
    produces_optional: tuple[ArtifactId, ...] = ()
    derives: tuple[ArtifactId, ...] = ()
    move: MachineMoveSpec | None = None
    query: ToolQuerySpec | None = None
    lower_context_id: str | None = None


CONTEXTS: tuple[ContextSpec, ...] = (
    ContextSpec(
        context_id="navigator",
        layer="navigator",
        label="Human/LLM navigator, web UI, SDK, or curl client",
    ),
    ContextSpec(
        context_id="action-registry",
        layer="registry",
        label="Transport-independent action contracts",
        parent_id="navigator",
    ),
    ContextSpec(
        context_id="episode-machine",
        layer="machine",
        label="Serialized artifact transition machine",
        parent_id="action-registry",
    ),
    ContextSpec(
        context_id="stage-0.ingestion",
        layer="delegated",
        label="Stage 0 ingestion file/code loop",
        parent_id="episode-machine",
        stage_id="stage-0",
        owns=("raw_data",),
        allowed_tools=("list_files", "read_file_sample", "execute_python", "submit_table"),
        runtime_state=("prepared_input_dir", "sandbox", "result_df", "column_descriptions"),
    ),
    ContextSpec(
        context_id="stage-1a.constructs",
        layer="delegated",
        label="Stage 1a construct proposal loop",
        parent_id="episode-machine",
        stage_id="stage-1a",
        owns=("constructs",),
        allowed_tools=("validate_latent_model",),
        runtime_state=("question", "latent_model_draft", "llm_trace"),
    ),
    ContextSpec(
        context_id="stage-1b.causal-model",
        layer="delegated",
        label="Stage 1b measurement, DAG, and identification loop",
        parent_id="episode-machine",
        stage_id="stage-1b",
        owns=("causal_spec", "identification_report"),
        allowed_tools=("validate_measurement_model",),
        runtime_state=("constructs", "dataset_schema", "causal_spec_draft", "identifiability"),
    ),
    ContextSpec(
        context_id="stage-2.measurement",
        layer="delegated",
        label="Stage 2 extraction worker fan-out",
        parent_id="episode-machine",
        stage_id="stage-2",
        owns=("extraction_report", "model_data"),
        allowed_tools=("validate_extractions",),
        runtime_state=("indicator_plan", "worker_statuses", "extracted_values"),
    ),
    ContextSpec(
        context_id="stage-3.validation",
        layer="delegated",
        label="Stage 3 measured-data validation",
        parent_id="episode-machine",
        stage_id="stage-3",
        owns=("validation_report",),
        runtime_state=("indicator_audits", "dataset_issues", "validation_status"),
    ),
    ContextSpec(
        context_id="stage-4.model-spec",
        layer="delegated",
        label="Stage 4 model/prior reducer",
        parent_id="episode-machine",
        stage_id="stage-4",
        owns=("compiled_ssm",),
        allowed_tools=("search_literature", "submit_model_spec", "submit_priors"),
        runtime_state=(
            "deterministic_skeleton",
            "immutable_plan",
            "cursor",
            "block_statuses",
            "accepted_state",
            "repair_campaign",
        ),
    ),
    ContextSpec(
        context_id="stage-5b.inference",
        layer="delegated",
        label="Stage 5b exact nonlinear SSM inference job",
        parent_id="episode-machine",
        stage_id="stage-5b",
        owns=("posterior",),
        runtime_state=("sampler_config", "diagnostics", "fitted_artifact"),
    ),
    ContextSpec(
        context_id="stage-6.ranking",
        layer="delegated",
        label="Stage 6 baseline causal ranking",
        parent_id="episode-machine",
        stage_id="stage-6",
        owns=("baseline_ranking",),
        allowed_tools=("get_model_info", "simulate"),
        runtime_state=("identified_treatments", "effect_summaries", "llm_trace"),
    ),
)


def _run_action(
    action_id: str,
    namespace: str,
    name: str,
    stage_id: str,
    *,
    mode: ActionMode,
    lower_context_id: str,
) -> ActionSpec:
    spec = stage_spec(stage_id)
    return ActionSpec(
        action_id=action_id,
        namespace=namespace,
        name=name,
        kind="produce",
        mode=mode,
        context_id="navigator",
        consumes=spec.consumes,
        produces=spec.produces,
        produces_optional=spec.produces_optional,
        derives=spec.derives,
        move=MachineMoveSpec(kind="run", stage_id=stage_id),
        lower_context_id=lower_context_id,
    )


ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(
        action_id="nav.state",
        namespace="nav",
        name="state",
        kind="read",
        mode="read",
        context_id="navigator",
    ),
    ActionSpec(
        action_id="nav.timeline",
        namespace="nav",
        name="timeline",
        kind="read",
        mode="read",
        context_id="navigator",
    ),
    ActionSpec(
        action_id="nav.events",
        namespace="nav",
        name="events",
        kind="read",
        mode="read",
        context_id="navigator",
    ),
    ActionSpec(
        action_id="nav.get",
        namespace="nav",
        name="get",
        kind="read",
        mode="read",
        context_id="navigator",
    ),
    ActionSpec(
        action_id="nav.versions",
        namespace="nav",
        name="versions",
        kind="read",
        mode="read",
        context_id="navigator",
    ),
    ActionSpec(
        action_id="nav.diff",
        namespace="nav",
        name="diff",
        kind="read",
        mode="read",
        context_id="navigator",
    ),
    ActionSpec(
        action_id="episode.create",
        namespace="episode",
        name="create",
        kind="produce",
        mode="direct",
        context_id="navigator",
        produces=("question",),
        move=MachineMoveSpec(kind="write", artifact_id="question"),
    ),
    ActionSpec(
        action_id="episode.attach_data",
        namespace="episode",
        name="attach_data",
        kind="external",
        mode="direct",
        context_id="navigator",
    ),
    _run_action(
        "episode.ingest_data",
        "episode",
        "ingest_data",
        "stage-0",
        mode="delegated",
        lower_context_id="stage-0.ingestion",
    ),
    ActionSpec(
        action_id="episode.refresh",
        namespace="episode",
        name="refresh",
        kind="driver",
        mode="async",
        context_id="navigator",
    ),
    _run_action(
        "specify.constructs",
        "specify",
        "constructs",
        "stage-1a",
        mode="delegated",
        lower_context_id="stage-1a.constructs",
    ),
    _run_action(
        "specify.model",
        "specify",
        "model",
        "stage-1b",
        mode="delegated",
        lower_context_id="stage-1b.causal-model",
    ),
    ActionSpec(
        action_id="specify.edit",
        namespace="specify",
        name="edit",
        kind="produce",
        mode="direct",
        context_id="navigator",
        consumes=("causal_spec",),
        produces=("causal_spec",),
        derives=("identification_report",),
        move=MachineMoveSpec(kind="write", artifact_id="causal_spec"),
    ),
    ActionSpec(
        action_id="specify.identify",
        namespace="specify",
        name="identify",
        kind="check",
        mode="direct",
        context_id="navigator",
        consumes=("causal_spec",),
        derives=("identification_report",),
    ),
    ActionSpec(
        action_id="specify.refine",
        namespace="specify",
        name="refine",
        kind="produce",
        mode="delegated",
        context_id="navigator",
        consumes=("causal_spec", "model_data", "validation_report"),
        produces=("causal_spec",),
        derives=("identification_report",),
        move=MachineMoveSpec(kind="write", artifact_id="causal_spec"),
        lower_context_id="stage-1b.causal-model",
    ),
    _run_action(
        "measure.extract",
        "measure",
        "extract",
        "stage-2",
        mode="delegated",
        lower_context_id="stage-2.measurement",
    ),
    _run_action(
        "analyze.validate",
        "analyze",
        "validate",
        "stage-3",
        mode="direct",
        lower_context_id="stage-3.validation",
    ),
    _run_action(
        "fit.compile",
        "fit",
        "compile",
        "stage-4",
        mode="delegated",
        lower_context_id="stage-4.model-spec",
    ),
    _run_action(
        "fit.infer",
        "fit",
        "infer",
        "stage-5b",
        mode="async",
        lower_context_id="stage-5b.inference",
    ),
    ActionSpec(
        action_id="fit.check",
        namespace="fit",
        name="check",
        kind="check",
        mode="direct",
        context_id="navigator",
        consumes=("compiled_ssm",),
    ),
    _run_action(
        "analyze.rank",
        "analyze",
        "rank",
        "stage-6",
        mode="delegated",
        lower_context_id="stage-6.ranking",
    ),
    ActionSpec(
        action_id="analyze.simulate",
        namespace="analyze",
        name="simulate",
        kind="query",
        mode="direct",
        context_id="navigator",
        consumes=("posterior", "causal_spec", "identification_report"),
        query=ToolQuerySpec(stage_id="stage-6", tool_name="simulate"),
    ),
    ActionSpec(
        action_id="analyze.counterfactual",
        namespace="analyze",
        name="counterfactual",
        kind="query",
        mode="direct",
        context_id="navigator",
        consumes=("posterior", "causal_spec", "identification_report"),
        query=ToolQuerySpec(stage_id="stage-6", tool_name="simulate"),
    ),
    ActionSpec(
        action_id="analyze.ppc",
        namespace="analyze",
        name="ppc",
        kind="check",
        mode="direct",
        context_id="navigator",
        consumes=("posterior", "model_data"),
    ),
    ActionSpec(
        action_id="analyze.save",
        namespace="analyze",
        name="save",
        kind="produce",
        mode="direct",
        context_id="navigator",
        consumes=("posterior",),
        produces=("saved_scenarios",),
        move=MachineMoveSpec(kind="write", artifact_id="saved_scenarios"),
    ),
)


CONTEXTS_BY_ID: dict[str, ContextSpec] = {context.context_id: context for context in CONTEXTS}
ACTIONS_BY_ID: dict[str, ActionSpec] = {action.action_id: action for action in ACTIONS}


def action_spec(action_id: str) -> ActionSpec:
    """Return one action spec by id."""
    try:
        return ACTIONS_BY_ID[action_id]
    except KeyError as exc:
        known = ", ".join(ACTIONS_BY_ID)
        raise KeyError(f"Unknown action '{action_id}'. Expected one of: {known}") from exc


def context_spec(context_id: str) -> ContextSpec:
    """Return one context spec by id."""
    try:
        return CONTEXTS_BY_ID[context_id]
    except KeyError as exc:
        known = ", ".join(CONTEXTS_BY_ID)
        raise KeyError(f"Unknown context '{context_id}'. Expected one of: {known}") from exc


def primary_stage_action(stage_id: str) -> ActionSpec:
    """Return the public action that runs a stage."""
    matches = [
        action
        for action in ACTIONS
        if action.move is not None
        and action.move.kind == "run"
        and action.move.stage_id == stage_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one primary action for {stage_id}, found {len(matches)}")
    return matches[0]


def action_is_enabled(state: EpisodeState, action: ActionSpec) -> bool:
    """Whether an action is worth surfacing (affordance, not machine legality).

    Reads are always available. A run is surfaced when every consumed artifact
    exists (this matches machine legality). A write/edit/query/check is surfaced
    when its referenced inputs exist — a stricter affordance guard than the
    always-legal ``write`` it may compile to.
    """
    if action.kind == "read":
        return True
    if action.move is not None and action.move.kind == "run":
        if action.move.stage_id is None:
            return False
        spec = stage_spec(action.move.stage_id)
        return all(state.has(artifact) for artifact in spec.consumes)
    return all(state.has(artifact) for artifact in action.consumes)


def legal_actions(state: EpisodeState) -> tuple[ActionSpec, ...]:
    """Public actions worth surfacing at the current artifact state."""
    return tuple(action for action in ACTIONS if action_is_enabled(state, action))


def legal_action_ids(state: EpisodeState) -> tuple[str, ...]:
    """Public action ids worth surfacing at the current artifact state."""
    return tuple(action.action_id for action in legal_actions(state))


def _move_dict(move: MachineMoveSpec | None) -> dict[str, str | None] | None:
    if move is None:
        return None
    return {
        "kind": move.kind,
        "stage_id": move.stage_id,
        "artifact_id": move.artifact_id,
    }


def _query_dict(query: ToolQuerySpec | None) -> dict[str, str | bool] | None:
    if query is None:
        return None
    return {
        "stage_id": query.stage_id,
        "tool_name": query.tool_name,
        "freshness_checked": query.freshness_checked,
    }


def describe_contexts() -> list[dict[str, object]]:
    """JSON-ready context registry description."""
    return [
        {
            "context_id": context.context_id,
            "layer": context.layer,
            "label": context.label,
            "parent_id": context.parent_id,
            "stage_id": context.stage_id,
            "owns": list(context.owns),
            "allowed_tools": list(context.allowed_tools),
            "runtime_state": list(context.runtime_state),
        }
        for context in CONTEXTS
    ]


def describe_actions() -> list[dict[str, object]]:
    """JSON-ready public action registry description."""
    return [
        {
            "action_id": action.action_id,
            "namespace": action.namespace,
            "name": action.name,
            "kind": action.kind,
            "mode": action.mode,
            "context_id": action.context_id,
            "consumes": list(action.consumes),
            "produces": list(action.produces),
            "produces_optional": list(action.produces_optional),
            "derives": list(action.derives),
            "move": _move_dict(action.move),
            "query": _query_dict(action.query),
            "lower_context_id": action.lower_context_id,
        }
        for action in ACTIONS
    ]


def _assert_hierarchy_consistent() -> None:
    stage_ids = {spec.stage_id for spec in ARTIFACT_GRAPH}
    artifact_ids = {
        artifact for spec in ARTIFACT_GRAPH for artifact in (*spec.consumes, *spec.all_produces)
    } | set(ROOT_ARTIFACTS)

    if len(CONTEXTS_BY_ID) != len(CONTEXTS):
        raise AssertionError("Context ids must be unique")
    if len(ACTIONS_BY_ID) != len(ACTIONS):
        raise AssertionError("Action ids must be unique")

    for context in CONTEXTS:
        if context.parent_id is not None and context.parent_id not in CONTEXTS_BY_ID:
            raise AssertionError(f"Unknown parent context '{context.parent_id}'")
        if context.stage_id is not None and context.stage_id not in stage_ids:
            raise AssertionError(f"Unknown context stage '{context.stage_id}'")
        unknown_owned = set(context.owns) - artifact_ids
        if unknown_owned:
            raise AssertionError(f"{context.context_id} owns unknown artifacts: {unknown_owned}")

    for stage_id in stage_ids:
        primary_stage_action(stage_id)

    for action in ACTIONS:
        if action.context_id not in CONTEXTS_BY_ID:
            raise AssertionError(f"{action.action_id} has unknown context {action.context_id}")
        if action.lower_context_id is not None and action.lower_context_id not in CONTEXTS_BY_ID:
            raise AssertionError(
                f"{action.action_id} has unknown lower context {action.lower_context_id}"
            )
        if action.move is not None and action.query is not None:
            raise AssertionError(f"{action.action_id} cannot have both move and query specs")
        if action.move is not None:
            if action.move.kind == "run":
                if action.move.stage_id not in stage_ids:
                    raise AssertionError(f"{action.action_id} runs unknown stage")
            elif action.move.kind == "write" and action.move.artifact_id not in WRITABLE_ARTIFACTS:
                raise AssertionError(f"{action.action_id} writes non-writable artifact")
        if action.query is not None and action.query.stage_id not in stage_ids:
            raise AssertionError(f"{action.action_id} queries unknown stage")
        referenced = (
            *action.consumes,
            *action.produces,
            *action.produces_optional,
            *action.derives,
        )
        if set(referenced) - artifact_ids:
            raise AssertionError(f"{action.action_id} references unknown artifacts")


_assert_hierarchy_consistent()
