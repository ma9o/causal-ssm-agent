"""Inspect AI evaluation for worker measurement instruction adherence.

Uses a judge model to evaluate how well competing worker models follow
the measurement instructions from the example DAG schema. The judge
ranks outputs without knowing model names and returns the winner.

Usage:
    inspect eval evals/worker_measurement_adherence.py --model openrouter/anthropic/claude-sonnet-4
"""

import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import random
import yaml

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver, system_message

from causal_agent.workers.prompts import WORKER_SYSTEM, WORKER_USER
from causal_agent.workers.agents import (
    _format_dimensions,
    _get_observed_dimension_dtypes,
    _get_outcome_description,
)
from causal_agent.utils.llm import make_worker_tools, multi_turn_generate, parse_json_response

from evals.common import (
    get_sample_chunks_worker,
    load_example_dag,
)


def load_eval_config() -> dict:
    """Load the eval config.yaml file."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# Load config
EVAL_CONFIG = load_eval_config()

# Worker models to compete
WORKER_MODELS = {m["id"]: m["alias"] for m in EVAL_CONFIG["worker_models"]}

# Questions from config
EVAL_QUESTIONS = EVAL_CONFIG["questions"]


JUDGE_SYSTEM = """\
You are an expert evaluator assessing data extraction quality. You will be shown:
1. A causal question
2. A schema with measurement instructions for each dimension
3. A data chunk
4. Multiple candidate extractions from different models (labeled A, B, C, etc.)

Your task is to rank the candidates from best to worst based on how well they follow the measurement instructions.

## Evaluation Criteria

For each candidate, assess:

1. **Instruction Adherence**: Did the model follow the `how_to_measure` instructions precisely?
   - Did it use the correct time thresholds (e.g., 22:00 for late_night, 20:00 for evening)?
   - Did it apply the correct matching rules (case-insensitive, correct keywords)?
   - Did it handle edge cases as specified (e.g., cross-midnight attribution)?

2. **Data Type Correctness**: Are values of the correct dtype?
   - binary: 0 or 1
   - count: non-negative integers
   - continuous: decimal numbers
   - categorical: strings

3. **Timestamp Granularity**: Are timestamps at the correct granularity?
   - daily: YYYY-MM-DD format
   - hourly: YYYY-MM-DDTHH:00 format

4. **Completeness**: Did the model extract data for all applicable dimensions present in the chunk?

5. **Accuracy**: When you can verify against the raw data, are the extractions correct?

## Output Format

Return a JSON object with your ranking:
```json
{
  "ranking": ["A", "B", "C"],
  "rationale": {
    "A": "Brief explanation of strengths/weaknesses",
    "B": "Brief explanation of strengths/weaknesses",
    "C": "Brief explanation of strengths/weaknesses"
  },
  "winner": "A"
}
```

The "ranking" array should list candidates from best to worst.
The "winner" field should contain the label of the best candidate.
"""

JUDGE_USER = """\
## Causal Question

{question}

## Measurement Instructions

{dimensions}

## Data Chunk

{chunk}

## Candidate Extractions

{candidates}

Please rank these candidates based on measurement instruction adherence and extraction quality.
"""


async def generate_worker_output(
    model_id: str,
    chunk: str,
    question: str,
    schema: dict,
) -> str:
    """Generate worker output for a single model.

    Returns the raw completion text (including JSON).
    """
    model = get_model(model_id)

    dimensions_text = _format_dimensions(schema)
    outcome_description = _get_outcome_description(schema)

    messages = [
        ChatMessageSystem(content=WORKER_SYSTEM),
        ChatMessageUser(
            content=WORKER_USER.format(
                question=question,
                outcome_description=outcome_description,
                dimensions=dimensions_text,
                chunk=chunk,
            )
        ),
    ]

    config = GenerateConfig(
        max_tokens=65536,
        reasoning_effort="high",
        reasoning_tokens=32768,
        reasoning_history="all",
    )

    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        tools=make_worker_tools(schema),
        config=config,
    )

    return completion


def format_candidates_for_judge(outputs: dict[str, str], label_map: dict[str, str]) -> str:
    """Format candidate outputs for the judge prompt.

    Args:
        outputs: Dict of model_id -> completion text
        label_map: Dict of model_id -> anonymous label (A, B, C, etc.)

    Returns:
        Formatted string with labeled candidates
    """
    parts = []
    for model_id, label in sorted(label_map.items(), key=lambda x: x[1]):
        output = outputs.get(model_id, "[ERROR: No output]")
        # Extract just the JSON part for cleaner comparison
        try:
            data = parse_json_response(output)
            json_str = json.dumps(data, indent=2)
        except Exception:
            json_str = output[:2000] + "..." if len(output) > 2000 else output

        parts.append(f"### Candidate {label}\n\n```json\n{json_str}\n```")

    return "\n\n".join(parts)


def create_eval_dataset(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset.

    Each sample contains:
    - A question from the eval set
    - A data chunk
    - Metadata with the question and chunk for worker generation

    Args:
        n_chunks: Number of chunks per question
        seed: Random seed for reproducibility
        input_file: Specific input file name, or None for latest

    Returns:
        MemoryDataset with samples
    """
    schema = load_example_dag()
    dimensions_text = _format_dimensions(schema)

    # Get chunks
    total_chunks = n_chunks * len(EVAL_QUESTIONS)
    chunks = get_sample_chunks_worker(total_chunks, seed, input_file)

    samples = []
    chunk_idx = 0

    for q in EVAL_QUESTIONS:
        for i in range(n_chunks):
            if chunk_idx >= len(chunks):
                break

            chunk = chunks[chunk_idx]
            chunk_idx += 1

            # The input is the judge prompt template - actual content filled in by solver
            samples.append(
                Sample(
                    input=f"Question: {q['question']}\nChunk index: {i}",
                    id=f"q{q['id']}_chunk{i}",
                    metadata={
                        "question_id": q["id"],
                        "question": q["question"],
                        "chunk": chunk,
                        "chunk_index": i,
                        "dimensions_text": dimensions_text,
                    },
                )
            )

    return MemoryDataset(samples)


def judge_solver(model_ids: list[str] | None = None):
    """Solver that generates worker outputs and asks judge to rank them.

    Args:
        model_ids: List of model IDs to compete. If None, uses all worker models.
    """
    if model_ids is None:
        model_ids = list(WORKER_MODELS.keys())

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            schema = load_example_dag()
            question = state.metadata["question"]
            chunk = state.metadata["chunk"]
            dimensions_text = state.metadata["dimensions_text"]

            # Generate outputs from all competing models in parallel
            tasks = {
                model_id: generate_worker_output(model_id, chunk, question, schema)
                for model_id in model_ids
            }

            outputs = {}
            for model_id, coro in tasks.items():
                try:
                    outputs[model_id] = await coro
                except Exception as e:
                    outputs[model_id] = f"[ERROR: {e}]"

            # Create anonymous labels and shuffle
            labels = [chr(ord("A") + i) for i in range(len(model_ids))]
            shuffled_models = model_ids.copy()
            random.seed(hash(state.sample_id))  # Deterministic shuffle per sample
            random.shuffle(shuffled_models)
            label_map = dict(zip(shuffled_models, labels))

            # Format candidates for judge
            candidates_text = format_candidates_for_judge(outputs, label_map)

            # Store label_map in metadata for scorer
            state.metadata["label_map"] = label_map
            state.metadata["reverse_label_map"] = {v: k for k, v in label_map.items()}

            # Build judge prompt
            judge_prompt = JUDGE_USER.format(
                question=question,
                dimensions=dimensions_text,
                chunk=chunk,
                candidates=candidates_text,
            )

            # Replace messages with judge prompt
            state.messages = [
                ChatMessageSystem(content=JUDGE_SYSTEM),
                ChatMessageUser(content=judge_prompt),
            ]

            # Generate judge response
            judge_model = get_model()
            config = GenerateConfig(
                max_tokens=4096,
                temperature=0.0,  # Deterministic judging
            )
            response = await judge_model.generate(state.messages, config=config)
            state.output.completion = response.completion

            return state

        return solve

    return _solver()


@scorer(metrics=[mean(), stderr()])
def measurement_adherence_scorer():
    """Score based on which model won the judge ranking.

    Returns:
        - The alias of the winning model as a categorical score
        - Score value is 1.0 if parsing succeeded, 0.0 otherwise
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        reverse_label_map = state.metadata.get("reverse_label_map", {})

        # Extract JSON from judge response
        try:
            # Find JSON in response
            import re
            json_match = re.search(r"\{[\s\S]*\}", completion)
            if not json_match:
                return Score(
                    value=0.0,
                    answer="[No JSON found in judge response]",
                    explanation=f"Judge response: {completion[:500]}...",
                )

            judge_data = json.loads(json_match.group())
            winner_label = judge_data.get("winner", "")
            ranking = judge_data.get("ranking", [])
            rationale = judge_data.get("rationale", {})

        except json.JSONDecodeError as e:
            return Score(
                value=0.0,
                answer="[JSON parse error]",
                explanation=f"Error: {e}\nResponse: {completion[:500]}...",
            )

        # Map winner label back to model
        winner_model = reverse_label_map.get(winner_label, "unknown")
        winner_alias = WORKER_MODELS.get(winner_model, winner_model)

        # Build ranking with model names
        ranking_with_names = []
        for label in ranking:
            model_id = reverse_label_map.get(label, "unknown")
            alias = WORKER_MODELS.get(model_id, model_id)
            ranking_with_names.append(f"{label}={alias}")

        explanation = (
            f"Winner: {winner_alias}\n"
            f"Full ranking: {', '.join(ranking_with_names)}\n"
            f"Rationale for winner: {rationale.get(winner_label, 'N/A')}"
        )

        return Score(
            value=1.0,  # Successfully parsed
            answer=winner_alias,
            explanation=explanation,
            metadata={
                "winner_model": winner_model,
                "winner_alias": winner_alias,
                "ranking": ranking,
                "ranking_with_names": ranking_with_names,
                "rationale": rationale,
            },
        )

    return score


@task
def worker_measurement_adherence_eval(
    n_chunks: int = 2,
    seed: int = 42,
    input_file: str | None = None,
    models: str | None = None,
):
    """Evaluate worker models on measurement instruction adherence.

    A judge model ranks competing worker outputs without knowing model names.
    Returns the winning model alias as the score.

    Args:
        n_chunks: Number of chunks per question (total samples = n_chunks * 5 questions)
        seed: Random seed for chunk sampling
        input_file: Specific preprocessed file name, or None for latest
        models: Comma-separated model IDs to compete, or None for all
    """
    # Parse models argument
    model_ids = None
    if models:
        model_ids = [m.strip() for m in models.split(",")]

    return Task(
        dataset=create_eval_dataset(n_chunks=n_chunks, seed=seed, input_file=input_file),
        solver=[
            judge_solver(model_ids=model_ids),
        ],
        scorer=measurement_adherence_scorer(),
    )
