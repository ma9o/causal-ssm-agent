"""Shared utilities for evals."""

import json
import re
from pathlib import Path

import yaml
from inspect_ai.model import get_model
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import Tool

from dsem_agent.utils.data import (
    DATA_DIR,
    PROCESSED_DIR,
    get_latest_preprocessed_file,
    get_orchestrator_chunk_size,
    get_worker_chunk_size,
    sample_chunks,
)
from dsem_agent.utils.llm import get_generate_config, multi_turn_generate


def load_eval_config() -> dict:
    """Load the eval config.yaml file."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_eval_questions() -> list[dict]:
    """Get evaluation questions from config."""
    return load_eval_config()["questions"]


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
        async def solve(state: TaskState, generate: Generate) -> TaskState:
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


def get_sample_chunks_orchestrator(n_chunks: int, seed: int, input_file: str | None = None) -> list[str]:
    """Get sampled chunks using orchestrator chunk size from config."""
    data_file = get_data_file(input_file)
    chunk_size = get_orchestrator_chunk_size()
    return sample_chunks(data_file, n_chunks, seed, chunk_size=chunk_size)


def get_sample_chunks_worker(n_chunks: int, seed: int, input_file: str | None = None) -> list[str]:
    """Get sampled chunks using worker chunk size from config."""
    data_file = get_data_file(input_file)
    chunk_size = get_worker_chunk_size()
    return sample_chunks(data_file, n_chunks, seed, chunk_size=chunk_size)


def extract_json_from_response(text: str) -> str | None:
    """Extract JSON from model response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(code_block_pattern, text)

    for match in matches:
        try:
            json.loads(match.strip())
            return match.strip()
        except json.JSONDecodeError:
            continue

    # Try to find raw JSON object
    brace_pattern = r"\{[\s\S]*\}"
    matches = re.findall(brace_pattern, text)

    for match in matches:
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue

    return None


def load_latent_model_by_question_id(question_id: int) -> dict:
    """Load a reference latent model by question ID.

    Args:
        question_id: The question ID (1-5) matching the latent model

    Returns:
        The latent model dict (constructs + edges)
    """
    config = load_eval_config()
    questions = config["questions"]
    question = next((q for q in questions if q["id"] == question_id), None)
    if question is None:
        raise ValueError(f"Question ID {question_id} not found in config")

    latent_path = DATA_DIR / question.get("latent", f"eval/latent_model{question_id}.json")
    with open(latent_path) as f:
        return json.load(f)


def load_causal_spec_by_question_id(question_id: int) -> dict:
    """Load a complete CausalSpec (latent + measurement) by question ID.

    Args:
        question_id: The question ID (1-5)

    Returns:
        The CausalSpec dict with 'latent' and 'measurement' keys
    """
    config = load_eval_config()
    questions = config["questions"]
    question = next((q for q in questions if q["id"] == question_id), None)
    if question is None:
        raise ValueError(f"Question ID {question_id} not found in config")

    causal_spec_path = DATA_DIR / question.get("dsem", f"eval/causal_spec{question_id}.json")
    with open(causal_spec_path) as f:
        return json.load(f)


