"""Inspect AI evaluation for worker measurement instruction adherence.

Uses a judge model to evaluate how well competing worker models follow
the measurement instructions from the CausalSpec schema. The judge
ranks outputs without knowing model names and returns the winner.

Uses the same core logic as production (via run_worker_extraction) for generating
worker outputs, just with different model configurations.

Usage:
    inspect eval evals/multi_model/eval3_worker_measurement_adherence.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval evals/multi_model/eval3_worker_measurement_adherence.py -T question=1
"""

import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import json
import random

from evals.common import (
    discover_questions,
    get_questions_with_causal_spec,
    get_sample_chunks_worker,
    load_eval_config,
    select_question,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from causal_ssm_agent.utils.llm import (
    get_generate_config,
    make_worker_generate_fn,
    parse_json_response,
)
from causal_ssm_agent.workers.core import (
    _format_indicators,
    _get_outcome_description,
    run_worker_extraction,
)
from causal_ssm_agent.workers.prompts.extraction import SYSTEM, USER

# Load config
_CONFIG = load_eval_config()

# Worker models to compete
WORKER_MODELS = {m["id"]: m["alias"] for m in _CONFIG["worker_models"]}


JUDGE_SYSTEM = """\
You are an expert evaluator assessing data extraction quality. You will be shown:
1. The exact prompt given to worker models (system and user messages)
2. Multiple candidate extractions from different models (labeled A, B, C, etc.)

Your task is to rank the candidates from best to worst based on how well they follow the instructions in the prompt.

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
## Worker Prompt

The following is the exact prompt given to the worker models:

### System Message

{system_prompt}

### User Message

{user_prompt}

## Candidate Extractions

{candidates}

Please rank these candidates based on how well they followed the instructions in the worker prompt.
"""


async def generate_worker_output(
    model_id: str,
    chunk: str,
    question: str,
    causal_spec: dict,
) -> str:
    """Generate worker output for a single model using core logic.

    Returns the raw JSON string of the output.
    """
    model = get_model(model_id)
    generate = make_worker_generate_fn(model)

    result = await run_worker_extraction(
        chunk=chunk,
        question=question,
        causal_spec=causal_spec,
        generate=generate,
    )

    return json.dumps(result.output.model_dump(), indent=2)


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
    question: str | None = None,
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset.

    Each sample contains:
    - A question from the eval set
    - A data chunk
    - Metadata with the full worker prompts for judge evaluation

    Args:
        question: Question selector for the CausalSpec to use. Defaults to first with causal_spec.
        n_chunks: Number of chunks per question
        seed: Random seed for reproducibility
        input_file: Specific input file name, or None for latest

    Returns:
        MemoryDataset with samples
    """
    # Resolve which CausalSpec to use for worker prompts
    available_cs = get_questions_with_causal_spec()
    if question:
        cs_question = select_question(available_cs, question)
    else:
        cs_question = available_cs[0]

    causal_spec = cs_question.load_causal_spec()
    indicators_text = _format_indicators(causal_spec)
    outcome_description = _get_outcome_description(causal_spec)

    # Iterate over all discovered questions for diverse prompts
    all_questions = discover_questions()

    # Get chunks
    total_chunks = n_chunks * len(all_questions)
    chunks = get_sample_chunks_worker(total_chunks, seed, input_file)

    samples = []
    chunk_idx = 0

    for q in all_questions:
        for i in range(n_chunks):
            if chunk_idx >= len(chunks):
                break

            chunk = chunks[chunk_idx]
            chunk_idx += 1

            # Build the full worker prompts that will be shown to the judge
            worker_user_prompt = USER.format(
                question=q.question,
                outcome_description=outcome_description,
                indicators=indicators_text,
                chunk=chunk,
            )

            # The input is the judge prompt template - actual content filled in by solver
            samples.append(
                Sample(
                    input=f"Question: {q.question}\nChunk index: {i}",
                    id=f"q_{q.slug}_chunk{i}",
                    metadata={
                        "question_slug": q.slug,
                        "question": q.question,
                        "chunk": chunk,
                        "chunk_index": i,
                        "causal_spec_slug": cs_question.slug,
                        "worker_system_prompt": SYSTEM,
                        "worker_user_prompt": worker_user_prompt,
                    },
                )
            )

    return MemoryDataset(samples)


def judge_solver(
    question: str | None = None,
    model_ids: list[str] | None = None,
    worker_timeout: float | None = None,
):
    """Solver that generates worker outputs and asks judge to rank them.

    Args:
        question: Question selector for CausalSpec. Defaults to first with causal_spec.
        model_ids: List of model IDs to compete. If None, uses all worker models.
        worker_timeout: Timeout in seconds for each worker. If None, uses config default.
    """
    if model_ids is None:
        model_ids = list(WORKER_MODELS.keys())

    # Get timeout from config if not specified
    if worker_timeout is None:
        worker_timeout = _CONFIG.get("worker_timeout_seconds", 180)

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            # Resolve the CausalSpec
            available_cs = get_questions_with_causal_spec()
            cs_slug = state.metadata.get("causal_spec_slug", question or "")
            cs_question = select_question(available_cs, cs_slug) if cs_slug else available_cs[0]
            causal_spec = cs_question.load_causal_spec()

            question_text = state.metadata["question"]
            chunk = state.metadata["chunk"]
            worker_system_prompt = state.metadata["worker_system_prompt"]
            worker_user_prompt = state.metadata["worker_user_prompt"]

            # Generate outputs from all competing models in parallel
            async def safe_generate(model_id: str) -> tuple[str, str]:
                """Generate with error handling and timeout, returns (model_id, result)."""
                try:
                    result = await asyncio.wait_for(
                        generate_worker_output(model_id, chunk, question_text, causal_spec),
                        timeout=worker_timeout,
                    )
                    return model_id, result
                except TimeoutError:
                    return model_id, f"[TIMEOUT: Worker did not finish within {worker_timeout}s]"
                except Exception as e:
                    return model_id, f"[ERROR: {e}]"

            results = await asyncio.gather(*[safe_generate(mid) for mid in model_ids])
            outputs = dict(results)

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

            # Build judge prompt with full worker prompts
            judge_prompt = JUDGE_USER.format(
                system_prompt=worker_system_prompt,
                user_prompt=worker_user_prompt,
                candidates=candidates_text,
            )

            # Replace messages with judge prompt
            state.messages = [
                ChatMessageSystem(content=JUDGE_SYSTEM),
                ChatMessageUser(content=judge_prompt),
            ]

            # Generate judge response
            judge_model = get_model()
            response = await judge_model.generate(state.messages, config=get_generate_config())
            state.output.completion = response.completion

            return state

        return solve

    return _solver()


@scorer(metrics=[mean(), stderr()])
def measurement_adherence_scorer():
    """Score based on judge ranking of model outputs.

    Returns:
        - Full ranking as "best > 2nd > 3rd > ..." in the answer field
        - Score value is 1.0 if parsing succeeded, 0.0 otherwise
    """

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
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
            ranking = judge_data.get("ranking", [])
            rationale = judge_data.get("rationale", {})

        except json.JSONDecodeError as e:
            return Score(
                value=0.0,
                answer="[JSON parse error]",
                explanation=f"Error: {e}\nResponse: {completion[:500]}...",
            )

        # Build ranking with model aliases
        ranking_aliases = []
        for label in ranking:
            model_id = reverse_label_map.get(label, "unknown")
            alias = WORKER_MODELS.get(model_id, model_id)
            ranking_aliases.append(alias)

        # Format as "1st > 2nd > 3rd > ..."
        ranking_str = " > ".join(ranking_aliases)

        # Build rationale summary
        rationale_parts = []
        for label in ranking:
            model_id = reverse_label_map.get(label, "unknown")
            alias = WORKER_MODELS.get(model_id, model_id)
            rationale_parts.append(f"{alias}: {rationale.get(label, 'N/A')}")

        explanation = "\n".join(rationale_parts)

        return Score(
            value=1.0,  # Successfully parsed
            answer=ranking_str,
            explanation=explanation,
            metadata={
                "ranking_aliases": ranking_aliases,
                "ranking_labels": ranking,
                "rationale": rationale,
            },
        )

    return score


@task
def worker_measurement_adherence_eval(
    question: str | None = None,
    n_chunks: int = 2,
    seed: int = 42,
    input_file: str | None = None,
    models: str | None = None,
    worker_timeout: int | None = None,
):
    """Evaluate worker models on measurement instruction adherence.

    A judge model ranks competing worker outputs without knowing model names.
    Returns the full ranking (e.g., "gemini > kimi > haiku") as the score answer.

    Args:
        question: Question selector for CausalSpec (prefix ID or slug). Defaults to first with causal_spec.
        n_chunks: Number of chunks per question (total samples = n_chunks * N questions)
        seed: Random seed for chunk sampling
        input_file: Specific preprocessed file name, or None for latest
        models: Comma-separated model IDs to compete, or None for all
        worker_timeout: Timeout in seconds for each worker (default: from config, 180s)
    """
    # Parse models argument
    model_ids = None
    if models:
        model_ids = [m.strip() for m in models.split(",")]

    return Task(
        dataset=create_eval_dataset(
            question=question,
            n_chunks=n_chunks,
            seed=seed,
            input_file=input_file,
        ),
        solver=[
            judge_solver(question=question, model_ids=model_ids, worker_timeout=worker_timeout),
        ],
        scorer=measurement_adherence_scorer(),
    )
