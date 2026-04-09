"""Shared utilities for evals."""

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
    execute_tools,
    get_model,
)
from inspect_ai.scorer import Score
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import Tool

# ══════════════════════════════════════════════════════════════════════════════
# Eval config (non-question settings)
# ══════════════════════════════════════════════════════════════════════════════


def load_eval_config() -> dict:
    """Load the eval config.yaml file."""
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)


EVAL_CONFIG = load_eval_config()
DEFAULT_EVAL_WORKSPACE_ID = str(EVAL_CONFIG.get("default_workspace_id", "GOLDEN"))


@dataclass(frozen=True)
class AnonymousLabelMapping:
    """Anonymous labels assigned to candidate IDs for a single sample."""

    label_map: dict[str, str]
    reverse_label_map: dict[str, str]


@dataclass(frozen=True)
class JudgeRanking:
    """Parsed judge ranking response."""

    ranking: list[str]
    rationale: dict[str, str]


def make_anonymous_label_mapping(sample_id: str, candidate_ids: list[str]) -> AnonymousLabelMapping:
    """Assign deterministic anonymous labels for judge-facing candidate sections."""
    labels = [chr(ord("A") + i) for i in range(len(candidate_ids))]
    shuffled_candidate_ids = candidate_ids.copy()
    random.seed(hash(sample_id))
    random.shuffle(shuffled_candidate_ids)
    label_map = dict(zip(shuffled_candidate_ids, labels))
    reverse_label_map = {label: candidate_id for candidate_id, label in label_map.items()}
    return AnonymousLabelMapping(label_map=label_map, reverse_label_map=reverse_label_map)


def format_labeled_candidates(
    mapping: AnonymousLabelMapping,
    render_candidate_body: Callable[[str], str],
) -> str:
    """Format anonymous candidate sections for a judge prompt."""
    sections = []
    for candidate_id, label in sorted(mapping.label_map.items(), key=lambda item: item[1]):
        body = render_candidate_body(candidate_id).strip()
        if body:
            sections.append(f"### Candidate {label}\n\n{body}")
        else:
            sections.append(f"### Candidate {label}")
    return "\n\n".join(sections)


def parse_judge_ranking_response(completion: str) -> JudgeRanking:
    """Extract ranking JSON from a judge completion."""
    json_match = re.search(r"\{[\s\S]*\}", completion)
    if not json_match:
        raise ValueError("No JSON found in judge response")

    try:
        judge_data = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}") from exc

    ranking = judge_data.get("ranking", [])
    rationale = judge_data.get("rationale", {})

    if not isinstance(ranking, list):
        raise ValueError("Judge response 'ranking' must be a list")
    if not isinstance(rationale, dict):
        raise ValueError("Judge response 'rationale' must be an object")

    return JudgeRanking(
        ranking=[str(label) for label in ranking],
        rationale={str(label): str(reason) for label, reason in rationale.items()},
    )


def score_judge_ranking_response(
    *,
    completion: str,
    reverse_label_map: dict[str, str],
    alias_lookup: dict[str, str],
    extra_metadata: dict[str, Any] | None = None,
) -> Score:
    """Build a standard ranking score from a parsed judge response."""
    try:
        ranking_result = parse_judge_ranking_response(completion)
    except ValueError as exc:
        message = str(exc)
        answer = (
            "[JSON parse error]"
            if message.startswith("JSON parse error:")
            else "[No JSON found in judge response]"
        )
        return Score(
            value=0.0,
            answer=answer,
            explanation=f"{message}\nResponse: {completion[:500]}...",
        )

    ranking_aliases = []
    for label in ranking_result.ranking:
        candidate_id = reverse_label_map.get(label, "unknown")
        ranking_aliases.append(alias_lookup.get(candidate_id, candidate_id))

    explanation = "\n".join(
        f"{alias_lookup.get(reverse_label_map.get(label, 'unknown'), reverse_label_map.get(label, 'unknown'))}: {ranking_result.rationale.get(label, 'N/A')}"
        for label in ranking_result.ranking
    )

    metadata: dict[str, Any] = {
        "ranking_aliases": ranking_aliases,
        "ranking_labels": ranking_result.ranking,
        "rationale": ranking_result.rationale,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return Score(
        value=1.0,
        answer=" > ".join(ranking_aliases),
        explanation=explanation,
        metadata=metadata,
    )


def parse_csv_task_arg(value: str | list[str] | None) -> list[str] | None:
    """Normalize Inspect task args that may arrive as CSV strings or lists."""
    if value is None:
        return None

    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(item.split(","))

    normalized = [part.strip() for part in parts if part.strip()]
    return normalized or None


# ══════════════════════════════════════════════════════════════════════════════
# Filesystem-driven question discovery
# ══════════════════════════════════════════════════════════════════════════════

EVAL_QUESTIONS_DIR = Path(__file__).parent / "questions"


@dataclass
class EvalQuestion:
    """An evaluation question discovered from the filesystem.

    Each question lives in ``evals/questions/<slug>/`` where slug is
    ``<id>_<short-name>`` (e.g. ``1_resolve-errors-faster``).
    """

    slug: str
    question: str
    dir: Path

    @property
    def id_prefix(self) -> str:
        """Numeric prefix, e.g. '1' from '1_resolve-errors-faster'."""
        return self.slug.split("_", 1)[0]

    # ── artifact checks ──

    @property
    def has_latent_model(self) -> bool:
        return (self.dir / "latent_model.json").exists()

    @property
    def has_causal_spec(self) -> bool:
        return (self.dir / "causal_spec.json").exists()

    @property
    def has_model_spec(self) -> bool:
        return (self.dir / "model_spec.json").exists()

    @property
    def has_priors(self) -> bool:
        return (self.dir / "priors.json").exists()

    @property
    def has_full_spec(self) -> bool:
        """Has model_spec + priors + causal_spec (all Stage 4 artifacts)."""
        return self.has_model_spec and self.has_priors and self.has_causal_spec

    # ── loaders ──

    def load_latent_model(self) -> dict:
        with (self.dir / "latent_model.json").open() as f:
            return json.load(f)

    def load_causal_spec(self) -> dict:
        with (self.dir / "causal_spec.json").open() as f:
            return json.load(f)

    def load_model_spec(self) -> dict:
        with (self.dir / "model_spec.json").open() as f:
            return json.load(f)

    def save_model_spec(self, spec: dict) -> Path:
        path = self.dir / "model_spec.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(spec, f, indent=2)
        return path

    def load_priors(self) -> dict:
        with (self.dir / "priors.json").open() as f:
            return json.load(f)

    def save_priors(self, priors: dict) -> Path:
        path = self.dir / "priors.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(priors, f, indent=2)
        return path


def discover_questions() -> list[EvalQuestion]:
    """Discover all eval questions from the filesystem.

    Globs ``evals/questions/*/question.yaml``, sorted by slug
    (numeric prefix gives natural order).
    """
    questions = []
    for qfile in sorted(EVAL_QUESTIONS_DIR.glob("*/question.yaml")):
        qdir = qfile.parent
        with qfile.open() as f:
            data = yaml.safe_load(f)
        questions.append(
            EvalQuestion(
                slug=qdir.name,
                question=data["question"],
                dir=qdir,
            )
        )
    return questions


def get_questions_with_latent_model() -> list[EvalQuestion]:
    """Return questions that have a latent_model.json artifact."""
    return [q for q in discover_questions() if q.has_latent_model]


def get_questions_with_causal_spec() -> list[EvalQuestion]:
    """Return questions that have a causal_spec.json artifact."""
    return [q for q in discover_questions() if q.has_causal_spec]


def get_questions_with_model_spec() -> list[EvalQuestion]:
    """Return questions that have a model_spec.json artifact."""
    return [q for q in discover_questions() if q.has_model_spec]


def get_questions_with_model_spec_and_causal_spec() -> list[EvalQuestion]:
    """Return questions that have both model_spec.json and causal_spec.json."""
    return [q for q in discover_questions() if q.has_model_spec and q.has_causal_spec]


def get_questions_with_full_spec() -> list[EvalQuestion]:
    """Return questions that have model_spec + priors + causal_spec."""
    return [q for q in discover_questions() if q.has_full_spec]


def select_question(questions: list[EvalQuestion], selector: str) -> EvalQuestion:
    """Select a question by numeric prefix or full slug."""
    for q in questions:
        if q.slug == selector or q.slug.startswith(f"{selector}_"):
            return q
    raise ValueError(f"No question matching '{selector}'. Available: {[q.slug for q in questions]}")


def select_questions(questions: list[EvalQuestion], selectors: str) -> list[EvalQuestion]:
    """Select multiple questions from a comma-separated selector string."""
    parts = [s.strip() for s in selectors.split(",")]
    return [select_question(questions, s) for s in parts]


# ══════════════════════════════════════════════════════════════════════════════
# Solvers & utilities
# ══════════════════════════════════════════════════════════════════════════════


def get_generate_config() -> GenerateConfig:
    """Build the standard Inspect GenerateConfig from project config."""
    from causal_ssm_agent.utils.config import get_config

    llm = get_config().llm
    return GenerateConfig(
        max_tokens=llm.max_tokens,
        timeout=llm.timeout,
        reasoning_effort=llm.reasoning_effort,
        reasoning_history="all",
    )


def _dict_messages_to_chat(messages: list[dict[str, Any]]) -> list[Any]:
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            chat_messages.append(ChatMessageSystem(content=msg["content"]))
        elif msg["role"] == "user":
            chat_messages.append(ChatMessageUser(content=msg["content"]))
    return chat_messages


async def multi_turn_generate(
    messages: list[Any],
    model: Model,
    follow_ups: list[str] | None = None,
    tools: list[Tool] | None = None,
    config: GenerateConfig | None = None,
) -> str:
    """Inspect-backed multi-turn generation for eval-only usage."""
    _config = config or GenerateConfig()
    working_messages = list(messages)
    follow_ups = follow_ups or []

    if tools:
        while True:
            output = await model.generate(input=working_messages, tools=tools, config=_config)
            working_messages.append(output.message)
            if output.message.tool_calls:
                tool_messages, tool_output = await execute_tools(
                    working_messages,
                    tools,
                    _config.max_tool_output,
                )
                working_messages.extend(tool_messages)
                if tool_output is not None:
                    output = tool_output
            if not output.message.tool_calls:
                break
        last_nonempty = output.completion
    else:
        output = await model.generate(working_messages, config=_config)
        working_messages.append(ChatMessageAssistant(content=output.completion))
        last_nonempty = output.completion

    for prompt in follow_ups:
        working_messages.append(ChatMessageUser(content=prompt))
        response = await model.generate(working_messages, config=_config)
        working_messages.append(ChatMessageAssistant(content=response.completion))
        if response.completion and response.completion.strip():
            last_nonempty = response.completion

    return last_nonempty


def make_generate_fn(
    model: Model,
    config: GenerateConfig | None = None,
):
    """Create an Inspect-backed generate function for core runtime logic."""
    _config = config or get_generate_config()

    async def generate(
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        follow_ups: list[str] | None = None,
        label: str | None = None,  # noqa: ARG001
    ) -> str:
        chat_messages = _dict_messages_to_chat(messages)
        if follow_ups or tools:
            return await multi_turn_generate(
                messages=chat_messages,
                model=model,
                follow_ups=follow_ups,
                tools=tools,
                config=_config,
            )
        response = await model.generate(chat_messages, config=_config)
        return response.completion

    return generate


def tool_assisted_generate(
    tools: list[Tool],
    follow_ups: list[str] | None = None,
):
    """Solver that runs multi-turn generation with tools.

    Uses multi_turn_generate with tools, ensuring evals test
    the exact same logic as production.

    Args:
        tools: List of tools available to the model
        follow_ups: Optional follow-up prompts after initial response
    """

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            model = get_model()
            config = get_generate_config()

            completion = await multi_turn_generate(
                messages=list(state.messages),
                model=model,
                follow_ups=follow_ups,
                tools=tools,
                config=config,
            )

            state.output.completion = completion
            return state

        return solve

    return _solver()


def resolve_eval_workspace_id(workspace_id: str | None = None) -> str:
    """Resolve the workspace used for eval inputs."""
    return workspace_id or DEFAULT_EVAL_WORKSPACE_ID


def sample_evenly[T](items: list[T], n: int, seed: int | None = None) -> list[T]:
    """Sample up to ``n`` items, spread across the full list with jitter."""
    if n <= 0 or not items:
        return []

    n = min(n, len(items))
    if n >= len(items):
        return list(items)

    rng = random.Random(seed)
    segment_size = len(items) / n
    sampled: list[T] = []
    for i in range(n):
        segment_start = int(i * segment_size)
        segment_end = int((i + 1) * segment_size)
        idx = rng.randint(segment_start, max(segment_start, segment_end - 1))
        sampled.append(items[idx])
    return sampled


def load_workspace_question(workspace_id: str | None = None) -> str:
    """Load the materialized query for an eval workspace."""
    from causal_ssm_agent.utils import storage
    from causal_ssm_agent.utils.data import DATA_URI

    resolved = resolve_eval_workspace_id(workspace_id)
    query_path = storage.join(DATA_URI, resolved, "query.txt")
    if not storage.exists(query_path):
        raise FileNotFoundError(f"No query.txt found for workspace '{resolved}'")
    return storage.read_text(query_path).strip()


def load_workspace_stage_state(stage_id: str, workspace_id: str | None = None) -> Any:
    """Restore a persisted stage contract for an eval workspace."""
    from causal_ssm_agent.flows.stage_registry import load_stage_state

    resolved = resolve_eval_workspace_id(workspace_id)
    return load_stage_state(resolved, stage_id)


def load_workspace_stage1b_inputs(workspace_id: str | None = None) -> dict[str, Any]:
    """Load the exact Stage 1b inputs from a persisted workspace run."""
    from causal_ssm_agent.flows.pipeline_helpers import format_schema_for_llm
    from causal_ssm_agent.flows.run_store import (
        STAGE0_PARQUET_FILENAMES,
        find_run_artifact,
        load_parquet,
    )

    resolved = resolve_eval_workspace_id(workspace_id)
    question = load_workspace_question(resolved)
    stage0 = load_workspace_stage_state("stage-0", resolved)
    stage1a = load_workspace_stage_state("stage-1a", resolved)

    raw_df_path = find_run_artifact(resolved, STAGE0_PARQUET_FILENAMES)
    raw_df = load_parquet(raw_df_path)
    column_descriptions = {c.name: c.description for c in stage0.column_descriptions}
    dataset_schema = format_schema_for_llm(raw_df, column_descriptions)

    return {
        "workspace_id": resolved,
        "question": question,
        "latent_model": stage1a.latent_model.model_dump(),
        "chunks": [dataset_schema],
        "dataset_summary": f"{raw_df.shape[0]} rows x {raw_df.shape[1]} columns",
    }


def load_workspace_stage2_inputs(workspace_id: str | None = None) -> dict[str, Any]:
    """Load the exact Stage 2 semantic-worker inputs from a persisted workspace run."""
    from causal_ssm_agent.flows.run_store import (
        STAGE0_PARQUET_FILENAMES,
        find_run_artifact,
        load_parquet,
    )
    from causal_ssm_agent.flows.stages.stage2_extract import _prepare_semantic_chunks
    from causal_ssm_agent.utils.config import get_config

    resolved = resolve_eval_workspace_id(workspace_id)
    question = load_workspace_question(resolved)
    stage1b = load_workspace_stage_state("stage-1b", resolved)

    raw_df_path = find_run_artifact(resolved, STAGE0_PARQUET_FILENAMES)
    raw_df = load_parquet(raw_df_path)
    causal_spec = stage1b.causal_spec.model_dump()
    semantic_inds = [
        indicator
        for indicator in causal_spec.get("measurement", {}).get("indicators", [])
        if indicator.get("extraction_mode", "semantic") == "semantic"
    ]
    if not semantic_inds:
        raise ValueError(f"Workspace '{resolved}' has no semantic indicators for Stage 2 evals")

    time_col = "timestamp"
    stage2_workers = get_config().stage2_workers
    model_clock = causal_spec.get("measurement", {}).get("model_clock", "1d")
    chunk_texts, chunk_window_starts, chunk_contexts = _prepare_semantic_chunks(
        raw_df=raw_df,
        semantic_inds=semantic_inds,
        causal_spec=causal_spec,
        model_clock=model_clock,
        time_col=time_col,
        windows_per_chunk=stage2_workers.windows_per_chunk,
        max_events_per_window=stage2_workers.max_events_per_window,
        max_windows=None,
    )
    if not chunk_texts:
        raise ValueError(f"Workspace '{resolved}' produced no Stage 2 semantic chunks")

    return {
        "workspace_id": resolved,
        "question": question,
        "causal_spec": causal_spec,
        "chunk_texts": chunk_texts,
        "chunk_window_starts": chunk_window_starts,
        "chunk_contexts": chunk_contexts,
    }


def get_stage2_eval_chunks(
    n_chunks: int,
    seed: int,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Sample Stage 2 semantic-worker chunks from a persisted workspace."""
    stage2_inputs = load_workspace_stage2_inputs(workspace_id)
    chunk_texts = sample_evenly(stage2_inputs["chunk_texts"], n_chunks, seed)
    return {
        **stage2_inputs,
        "sampled_chunk_texts": chunk_texts,
    }
