"""Shared utilities for evals."""

import json
import logging
import random
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
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
)
from inspect_ai.scorer import Score
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
DEFAULT_EVAL_WORKSPACE_ID = str(EVAL_CONFIG.get("default_workspace_id", "DEMO"))


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
    label_map = dict(zip(shuffled_candidate_ids, labels, strict=True))
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

EVAL_QUESTIONS_DIR = Path(__file__).parent.parent / "data" / "questions"


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
    def has_latent_structure(self) -> bool:
        return (self.dir / "latent_structure.json").exists()

    @property
    def has_causal_design(self) -> bool:
        return (self.dir / "causal_design.json").exists()

    @property
    def has_statistical_model_spec(self) -> bool:
        return (self.dir / "statistical_model_spec.json").exists()

    @property
    def has_priors(self) -> bool:
        return (self.dir / "priors.json").exists()

    @property
    def has_full_spec(self) -> bool:
        """Has statistical_model_spec + priors + causal_design (all Target 4 artifacts)."""
        return self.has_statistical_model_spec and self.has_priors and self.has_causal_design


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
    from nof1_causal_lab.utils.config import get_config

    embedded = get_config().llm.embedded
    return GenerateConfig(
        max_tokens=embedded.max_tokens,
        timeout=embedded.timeout,
        reasoning_effort=embedded.reasoning_effort,
        reasoning_history="all",
    )


def make_eval_session_factory(
    context_id: str, model: str | None = None, *, max_tool_turns: int = 40
):
    """Open a real ``ScopedSessionFactory`` for the model under test.

    Returns an async context manager, which yields a
    :class:`~nof1_causal_lab.utils.agent_session.ScopedSessionFactory` bound to
    the project's embedded OpenRouter backend — the same path production uses, so
    the eval exercises the live target code rather than an Inspect-mediated copy.

    ``model`` defaults to the configured Target 1 model and must be an
    ``openrouter/...`` slug. Relies on ``OPENROUTER_API_KEY`` in the environment.
    """
    from nof1_causal_lab.utils.agent_session import ScopedSessionFactory
    from nof1_causal_lab.utils.config import LLMProfileConfig, get_config

    config = get_config()
    resolved_model = model or config.structure_proposal.llm.model
    logger = logging.getLogger(f"nof1_causal_lab.evals.{context_id}")

    @asynccontextmanager
    async def _open():
        started_at = time.monotonic()
        logger.info("[%s] starting", context_id)
        factory = ScopedSessionFactory(
            LLMProfileConfig(harness="none", model=resolved_model),
            config.llm,
            context_id=context_id,
            max_tool_turns=max_tool_turns,
        )
        try:
            yield factory
        except Exception:
            logger.exception("[%s] failed after %.1fs", context_id, time.monotonic() - started_at)
            raise
        logger.info("[%s] completed in %.1fs", context_id, time.monotonic() - started_at)

    return _open()


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


def _workspace_store(workspace_id: str) -> tuple[Any, Any]:
    """Artifact store plus current episode state for an eval workspace."""
    from nof1_causal_lab.machine.store import ArtifactStore, derive_current_state

    return ArtifactStore(workspace_id), derive_current_state(workspace_id)


def _current_version(state: Any, workspace_id: str, artifact_id: str) -> int:
    info = state.get(artifact_id)
    if info is None:
        raise FileNotFoundError(
            f"No current '{artifact_id}' artifact for workspace '{workspace_id}'"
        )
    return info.version


def load_workspace_question(workspace_id: str | None = None) -> str:
    """Load the current question artifact for an eval workspace."""
    resolved = resolve_eval_workspace_id(workspace_id)
    store, state = _workspace_store(resolved)
    version = _current_version(state, resolved, "question")
    return store.read_json_file("question", version, "question.json")["text"]


def load_workspace_measurement_structure_inputs(workspace_id: str | None = None) -> dict[str, Any]:
    """Load the exact Target 1b inputs from a workspace's artifact store."""
    from nof1_causal_lab.flows.pipeline_helpers import format_schema_for_llm

    resolved = resolve_eval_workspace_id(workspace_id)
    question = load_workspace_question(resolved)
    store, state = _workspace_store(resolved)

    raw_version = _current_version(state, resolved, "raw_data")
    profile = store.read_json_file("raw_data", raw_version, "profile.json")
    raw_df = store.read_parquet_file("raw_data", raw_version, "raw.parquet")
    constructs = store.read_json_file(
        "constructs", _current_version(state, resolved, "constructs"), "constructs.json"
    )

    column_descriptions = {
        column["name"]: column["description"] for column in profile.get("column_descriptions", [])
    }
    dataset_schema = format_schema_for_llm(raw_df, column_descriptions)

    return {
        "workspace_id": resolved,
        "question": question,
        "latent_structure": constructs["latent_structure"],
        "chunks": [dataset_schema],
        "dataset_summary": f"{raw_df.shape[0]} rows x {raw_df.shape[1]} columns",
    }


def load_workspace_extraction_inputs(workspace_id: str | None = None) -> dict[str, Any]:
    """Load the exact Target 2 semantic-worker inputs from a workspace's artifact store."""
    from nof1_causal_lab.flows.transitions.extraction.planning import prepare_semantic_chunks
    from nof1_causal_lab.utils.config import get_config

    resolved = resolve_eval_workspace_id(workspace_id)
    question = load_workspace_question(resolved)
    store, state = _workspace_store(resolved)

    raw_df = store.read_parquet_file(
        "raw_data", _current_version(state, resolved, "raw_data"), "raw.parquet"
    )
    causal_design = store.read_json_file(
        "causal_design", _current_version(state, resolved, "causal_design"), "causal_design.json"
    )["causal_design"]
    semantic_inds = [
        indicator
        for indicator in causal_design.get("measurement", {}).get("indicators", [])
        if indicator.get("extraction_mode", "semantic") == "semantic"
    ]
    if not semantic_inds:
        raise ValueError(f"Workspace '{resolved}' has no semantic indicators for Target 2 evals")

    time_col = "timestamp"
    extraction_workers = get_config().extraction_workers
    model_clock = causal_design.get("measurement", {}).get("model_clock", "1d")
    chunk_texts, chunk_window_starts, chunk_contexts = prepare_semantic_chunks(
        raw_df=raw_df,
        semantic_inds=semantic_inds,
        causal_design=causal_design,
        model_clock=model_clock,
        time_col=time_col,
        windows_per_chunk=extraction_workers.windows_per_chunk,
        max_events_per_window=extraction_workers.max_events_per_window,
        max_windows=None,
    )
    if not chunk_texts:
        raise ValueError(f"Workspace '{resolved}' produced no Target 2 semantic chunks")

    return {
        "workspace_id": resolved,
        "question": question,
        "causal_design": causal_design,
        "chunk_texts": chunk_texts,
        "chunk_window_starts": chunk_window_starts,
        "chunk_contexts": chunk_contexts,
    }


def get_extraction_eval_chunks(
    n_chunks: int,
    seed: int,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Sample Target 2 semantic-worker chunks from a persisted workspace."""
    extraction_inputs = load_workspace_extraction_inputs(workspace_id)
    chunk_texts = sample_evenly(extraction_inputs["chunk_texts"], n_chunks, seed)
    return {
        **extraction_inputs,
        "sampled_chunk_texts": chunk_texts,
    }
