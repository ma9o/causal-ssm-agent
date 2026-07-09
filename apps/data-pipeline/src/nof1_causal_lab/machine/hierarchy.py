"""Public naming and context layer above the artifact machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.machine.artifacts import ARTIFACT_IDS
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    DERIVATIONS,
    ROOT_ARTIFACTS,
    WRITABLE_ARTIFACTS,
    transition_spec,
)

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import ArtifactId, EpisodeState


ContextLayer = Literal["navigator", "registry", "machine", "delegated", "tool"]
ActionKind = Literal["read", "produce", "check", "query", "driver", "external"]
ActionMode = Literal["direct", "delegated", "async", "read"]
MoveKind = Literal["run", "write"]


@dataclass(frozen=True)
class ContextSpec:
    context_id: str
    layer: ContextLayer
    label: str
    parent_id: str | None = None
    owns: tuple[ArtifactId, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    runtime_state: tuple[str, ...] = ()


@dataclass(frozen=True)
class MachineMoveSpec:
    kind: MoveKind
    artifact_id: ArtifactId


@dataclass(frozen=True)
class ToolQuerySpec:
    context_id: str
    tool_name: str
    freshness_checked: bool = True


@dataclass(frozen=True)
class ActionSpec:
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
        context_id="ingestion",
        layer="delegated",
        label="Ingestion file/code loop",
        parent_id="episode-machine",
        owns=("raw_data",),
        allowed_tools=("list_files", "read_file_sample", "execute_python", "submit_table"),
        runtime_state=("prepared_input_dir", "sandbox", "result_df", "column_descriptions"),
    ),
    ContextSpec(
        context_id="latent-structure",
        layer="delegated",
        label="Latent structure proposal loop",
        parent_id="episode-machine",
        owns=("latent_structure",),
        allowed_tools=("validate_latent_structure",),
        runtime_state=("question", "latent_structure_draft", "llm_trace_ref"),
    ),
    ContextSpec(
        context_id="measurement-structure",
        layer="delegated",
        label="Measurement structure proposal loop",
        parent_id="episode-machine",
        owns=("measurement_structure",),
        allowed_tools=("validate_measurement_structure",),
        runtime_state=("latent_structure", "dataset_schema", "measurement_structure_draft"),
    ),
    ContextSpec(
        context_id="measurement",
        layer="delegated",
        label="Indicator extraction worker fan-out",
        parent_id="episode-machine",
        owns=("measurements", "panel"),
        allowed_tools=("validate_extractions",),
        runtime_state=("indicator_plan", "worker_statuses", "extracted_values"),
    ),
    ContextSpec(
        context_id="statistical-model-spec",
        layer="delegated",
        label="Model/prior reducer",
        parent_id="episode-machine",
        owns=("statistical_model_spec",),
        allowed_tools=("search_literature", "submit_statistical_model_spec", "submit_priors"),
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
        context_id="inference",
        layer="delegated",
        label="Exact nonlinear SSM inference job",
        parent_id="episode-machine",
        owns=("posterior",),
        runtime_state=("sampler_config", "diagnostics", "fitted_artifact"),
    ),
    ContextSpec(
        context_id="ranking",
        layer="delegated",
        label="Baseline causal ranking",
        parent_id="episode-machine",
        owns=("baseline_report",),
        allowed_tools=("get_model_info", "simulate"),
        runtime_state=("identified_treatments", "effect_summaries", "llm_trace_ref"),
    ),
)


def _reachable_derivations(artifact_id: ArtifactId) -> tuple[ArtifactId, ...]:
    found: list[ArtifactId] = []
    frontier = [artifact_id]
    while frontier:
        parent = frontier.pop(0)
        for spec in DERIVATIONS:
            if parent not in spec.from_ or spec.produces in found:
                continue
            found.append(spec.produces)
            frontier.append(spec.produces)
    return tuple(found)


def _run_action(
    action_id: str,
    namespace: str,
    name: str,
    artifact_id: ArtifactId,
    *,
    mode: ActionMode,
    lower_context_id: str,
) -> ActionSpec:
    spec = transition_spec(artifact_id)
    return ActionSpec(
        action_id=action_id,
        namespace=namespace,
        name=name,
        kind="produce",
        mode=mode,
        context_id="navigator",
        consumes=spec.consumes,
        produces=(spec.produces,),
        produces_optional=spec.produces_optional,
        derives=_reachable_derivations(spec.transition_id),
        move=MachineMoveSpec(kind="run", artifact_id=artifact_id),
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
        "raw_data",
        mode="delegated",
        lower_context_id="ingestion",
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
        "specify.latent_structure",
        "specify",
        "latent_structure",
        "latent_structure",
        mode="delegated",
        lower_context_id="latent-structure",
    ),
    _run_action(
        "specify.measurement",
        "specify",
        "measurement",
        "measurement_structure",
        mode="delegated",
        lower_context_id="measurement-structure",
    ),
    ActionSpec(
        action_id="specify.edit",
        namespace="specify",
        name="edit",
        kind="produce",
        mode="direct",
        context_id="navigator",
        consumes=("measurement_structure",),
        produces=("measurement_structure",),
        derives=_reachable_derivations("measurement_structure"),
        move=MachineMoveSpec(kind="write", artifact_id="measurement_structure"),
    ),
    ActionSpec(
        action_id="specify.identify",
        namespace="specify",
        name="identify",
        kind="check",
        mode="direct",
        context_id="navigator",
        consumes=("causal_design",),
        derives=("identification_report",),
    ),
    _run_action(
        "measure.extract",
        "measure",
        "extract",
        "measurements",
        mode="delegated",
        lower_context_id="measurement",
    ),
    _run_action(
        "fit.specify",
        "fit",
        "specify",
        "statistical_model_spec",
        mode="delegated",
        lower_context_id="statistical-model-spec",
    ),
    _run_action(
        "fit.infer",
        "fit",
        "infer",
        "posterior",
        mode="async",
        lower_context_id="inference",
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
        "baseline_report",
        mode="delegated",
        lower_context_id="ranking",
    ),
    ActionSpec(
        action_id="analyze.simulate",
        namespace="analyze",
        name="simulate",
        kind="query",
        mode="direct",
        context_id="navigator",
        consumes=("posterior", "causal_design", "identification_report"),
        query=ToolQuerySpec(context_id="ranking", tool_name="simulate"),
    ),
    ActionSpec(
        action_id="analyze.counterfactual",
        namespace="analyze",
        name="counterfactual",
        kind="query",
        mode="direct",
        context_id="navigator",
        consumes=("posterior", "causal_design", "identification_report"),
        query=ToolQuerySpec(context_id="ranking", tool_name="simulate"),
    ),
    ActionSpec(
        action_id="analyze.ppc",
        namespace="analyze",
        name="ppc",
        kind="check",
        mode="direct",
        context_id="navigator",
        consumes=("posterior", "panel"),
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
    try:
        return ACTIONS_BY_ID[action_id]
    except KeyError as exc:
        known = ", ".join(ACTIONS_BY_ID)
        raise KeyError(f"Unknown action '{action_id}'. Expected one of: {known}") from exc


def context_spec(context_id: str) -> ContextSpec:
    try:
        return CONTEXTS_BY_ID[context_id]
    except KeyError as exc:
        known = ", ".join(CONTEXTS_BY_ID)
        raise KeyError(f"Unknown context '{context_id}'. Expected one of: {known}") from exc


def primary_transition_action(artifact_id: ArtifactId) -> ActionSpec:
    matches = [
        action
        for action in ACTIONS
        if action.move is not None
        and action.move.kind == "run"
        and action.move.artifact_id == artifact_id
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one primary action for {artifact_id}, found {len(matches)}"
        )
    return matches[0]


def action_is_enabled(state: EpisodeState, action: ActionSpec) -> bool:
    """Affordance-level enabledness for navigator actions.

    This is intentionally not the machine legality engine. Run actions delegate
    to the transition existence guard; read/query/check actions may add
    usefulness preconditions so the navigator does not surface actions with no
    meaningful context.
    """
    if action.kind == "read":
        return True
    if action.move is not None and action.move.kind == "run":
        spec = transition_spec(action.move.artifact_id)
        return all(state.has(artifact) for artifact in spec.consumes)
    return all(state.has(artifact) for artifact in action.consumes)


def legal_actions(state: EpisodeState) -> tuple[ActionSpec, ...]:
    return tuple(action for action in ACTIONS if action_is_enabled(state, action))


def legal_action_ids(state: EpisodeState) -> tuple[str, ...]:
    return tuple(action.action_id for action in legal_actions(state))


def _move_dict(move: MachineMoveSpec | None) -> dict[str, str] | None:
    if move is None:
        return None
    return {"kind": move.kind, "artifact_id": move.artifact_id}


def _query_dict(query: ToolQuerySpec | None) -> dict[str, str | bool] | None:
    if query is None:
        return None
    return {
        "context_id": query.context_id,
        "tool_name": query.tool_name,
        "freshness_checked": query.freshness_checked,
    }


def describe_contexts() -> list[dict[str, object]]:
    return [
        {
            "context_id": context.context_id,
            "layer": context.layer,
            "label": context.label,
            "parent_id": context.parent_id,
            "owns": list(context.owns),
            "allowed_tools": list(context.allowed_tools),
            "runtime_state": list(context.runtime_state),
        }
        for context in CONTEXTS
    ]


def describe_actions() -> list[dict[str, object]]:
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
    transition_ids = {spec.transition_id for spec in ARTIFACT_GRAPH}
    artifact_ids = set(ARTIFACT_IDS)

    if len(CONTEXTS_BY_ID) != len(CONTEXTS):
        raise AssertionError("Context ids must be unique")
    if len(ACTIONS_BY_ID) != len(ACTIONS):
        raise AssertionError("Action ids must be unique")

    for context in CONTEXTS:
        if context.parent_id is not None and context.parent_id not in CONTEXTS_BY_ID:
            raise AssertionError(f"Unknown parent context '{context.parent_id}'")
        unknown_owned = set(context.owns) - artifact_ids
        if unknown_owned:
            raise AssertionError(f"{context.context_id} owns unknown artifacts: {unknown_owned}")

    for artifact_id in transition_ids:
        primary_transition_action(artifact_id)

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
            if action.move.kind == "run" and action.move.artifact_id not in transition_ids:
                raise AssertionError(f"{action.action_id} runs unknown transition")
            if action.move.kind == "write" and action.move.artifact_id not in WRITABLE_ARTIFACTS:
                raise AssertionError(f"{action.action_id} writes non-writable artifact")
        if action.query is not None and action.query.context_id not in CONTEXTS_BY_ID:
            raise AssertionError(f"{action.action_id} queries unknown context")
        referenced = (
            *action.consumes,
            *action.produces,
            *action.produces_optional,
            *action.derives,
        )
        if set(referenced) - artifact_ids:
            raise AssertionError(f"{action.action_id} references unknown artifacts")

    writable_produced = {spec.transition_id for spec in ARTIFACT_GRAPH if spec.writable}
    if set(WRITABLE_ARTIFACTS) != set(ROOT_ARTIFACTS) | writable_produced:
        raise AssertionError("Writable surface must be roots plus writable transitions")


_assert_hierarchy_consistent()
