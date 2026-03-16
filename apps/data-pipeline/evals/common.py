"""Shared utilities for evals."""

import json
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
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import Tool

from causal_ssm_agent.utils.data import (
    PROCESSED_DIR,
    get_latest_preprocessed_file,
    get_orchestrator_chunk_size,
    get_worker_chunk_size,
    sample_chunks,
)

# ══════════════════════════════════════════════════════════════════════════════
# Eval config (non-question settings)
# ══════════════════════════════════════════════════════════════════════════════


def load_eval_config() -> dict:
    """Load the eval config.yaml file."""
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)


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


# Files to exclude when finding the latest data file (script outputs)
EXCLUDE_FILES = {"orchestrator-samples-manual.txt"}


def format_chunks(chunks: list[str]) -> str:
    """Format chunks for prompts."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"--- CHUNK {i + 1} ---\n{chunk}")
    return "\n\n".join(parts)


def get_data_file(input_file: str | None = None):
    """Resolve data file path."""
    if input_file:
        data_file = PROCESSED_DIR / input_file
        if not data_file.exists():
            raise FileNotFoundError(f"File not found: {data_file}")
        return data_file

    data_file = get_latest_preprocessed_file(exclude=EXCLUDE_FILES)
    if not data_file:
        raise FileNotFoundError(f"No data files found in {PROCESSED_DIR}")
    return data_file


def get_sample_chunks_orchestrator(
    n_chunks: int, seed: int, input_file: str | None = None
) -> list[str]:
    """Get sampled chunks using orchestrator chunk size from config."""
    data_file = get_data_file(input_file)
    chunk_size = get_orchestrator_chunk_size()
    return sample_chunks(data_file, n_chunks, seed, chunk_size=chunk_size)


def get_sample_chunks_worker(n_chunks: int, seed: int, input_file: str | None = None) -> list[str]:
    """Get sampled chunks using worker chunk size from config."""
    data_file = get_data_file(input_file)
    chunk_size = get_worker_chunk_size()
    return sample_chunks(data_file, n_chunks, seed, chunk_size=chunk_size)
