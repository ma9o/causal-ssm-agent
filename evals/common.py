"""Shared utilities for evals."""

import json
import re

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import Generate, TaskState, generate, solver

from causal_agent.orchestrator.agents import run_two_stage_proposal
from causal_agent.utils.data import (
    PROCESSED_DIR,
    get_latest_preprocessed_file,
    sample_chunks,
    get_orchestrator_chunk_size,
    get_worker_chunk_size,
)


def reasoning_generate():
    """Generate solver with max thinking budget for reasoning models."""
    return generate(
        max_tokens=65536,
        reasoning_effort="high",
    )


@solver
def two_stage_proposal_solver():
    """Solver that runs the production two-stage proposal pipeline.

    Uses run_two_stage_proposal from agents.py directly, ensuring
    evals test the exact same logic as production.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get the model being evaluated
        model = get_model()

        # Config for reasoning models
        config = GenerateConfig(
            max_tokens=65536,
            reasoning_effort="high",
        )

        # Run the production two-stage pipeline
        # state.messages already has system + user from the task setup
        completion = await run_two_stage_proposal(
            messages=list(state.messages),
            model=model,
            config=config,
        )

        # Update state with final completion
        # Note: state.messages won't reflect the multi-turn, but state.output will have the final result
        state.output.completion = completion

        return state

    return solve

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


def load_example_dag() -> dict:
    """Load the example DAG for worker evals."""
    dag_file = PROCESSED_DIR.parent / "eval" / "example_dag.json"
    with open(dag_file) as f:
        return json.load(f)
