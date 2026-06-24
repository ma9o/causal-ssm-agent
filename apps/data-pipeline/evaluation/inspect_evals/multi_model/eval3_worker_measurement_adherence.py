"""Inspect AI evaluation for worker measurement instruction adherence.

Uses a judge model to evaluate how well competing worker models follow
the measurement instructions from the CausalSpec schema. The judge
ranks outputs without knowing model names and returns the winner.

Uses the same core logic as production (via run_worker_extraction) for generating
worker outputs, just with different model configurations.

Usage:
    inspect eval evals/multi_model/eval3_worker_measurement_adherence.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval evals/multi_model/eval3_worker_measurement_adherence.py -T workspace_id=SMALLGOLDEN
"""

import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
import json

from evaluation.inspect_evals.common import (
    format_labeled_candidates,
    get_generate_config,
    get_stage2_eval_chunks,
    load_eval_config,
    make_anonymous_label_mapping,
    make_generate_fn,
    parse_csv_task_arg,
    score_judge_ranking_response,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from nof1_causal_lab.utils.llm import parse_json_response
from nof1_causal_lab.workers.core import (
    _format_indicators,
    _get_outcome_description,
    run_worker_extraction,
)
from nof1_causal_lab.workers.prompts.extraction import SYSTEM, USER

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
    generate = make_generate_fn(model)

    result = await run_worker_extraction(
        chunk=chunk,
        question=question,
        causal_spec=causal_spec,
        generate=generate,
    )

    return json.dumps(result.output.model_dump(), indent=2)


def create_eval_dataset(
    n_chunks: int = 5,
    seed: int = 42,
    workspace_id: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset.

    Each sample contains:
    - A workspace question
    - A data chunk
    - Metadata with the full worker prompts for judge evaluation

    Args:
        n_chunks: Number of chunks per question
        seed: Random seed for reproducibility
        workspace_id: Workspace to load persisted Stage 2 inputs from.

    Returns:
        MemoryDataset with samples
    """
    stage2_inputs = get_stage2_eval_chunks(n_chunks, seed, workspace_id)
    causal_spec = stage2_inputs["causal_spec"]
    indicators_text = _format_indicators(causal_spec)
    outcome_description = _get_outcome_description(causal_spec)

    samples = []
    for i, chunk in enumerate(stage2_inputs["sampled_chunk_texts"]):
        worker_user_prompt = USER.format(
            question=stage2_inputs["question"],
            outcome_description=outcome_description,
            indicators=indicators_text,
            chunk=chunk,
        )

        samples.append(
            Sample(
                input=f"Workspace: {stage2_inputs['workspace_id']}\nChunk index: {i}",
                id=f"workspace_{stage2_inputs['workspace_id']}_chunk{i}",
                metadata={
                    "workspace_id": stage2_inputs["workspace_id"],
                    "question": stage2_inputs["question"],
                    "chunk": chunk,
                    "chunk_index": i,
                    "causal_spec": causal_spec,
                    "worker_system_prompt": SYSTEM,
                    "worker_user_prompt": worker_user_prompt,
                },
            )
        )

    return MemoryDataset(samples)


def judge_solver(
    model_ids: list[str] | None = None,
    worker_timeout: float | None = None,
):
    """Solver that generates worker outputs and asks judge to rank them.

    Args:
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
            causal_spec = state.metadata["causal_spec"]
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
                except Exception as e:  # noqa: BLE001
                    return model_id, f"[ERROR: {e}]"

            results = await asyncio.gather(*[safe_generate(mid) for mid in model_ids])
            outputs = dict(results)

            # Create anonymous labels and shuffle
            label_mapping = make_anonymous_label_mapping(
                sample_id=state.sample_id,
                candidate_ids=model_ids,
            )

            # Format candidates for judge
            def _render_candidate(model_id: str) -> str:
                output = outputs.get(model_id, "[ERROR: No output]")
                try:
                    data = parse_json_response(output)
                    json_str = json.dumps(data, indent=2)
                except Exception:  # noqa: BLE001
                    json_str = output[:2000] + "..." if len(output) > 2000 else output
                return f"```json\n{json_str}\n```"

            candidates_text = format_labeled_candidates(label_mapping, _render_candidate)

            # Store label_map in metadata for scorer
            state.metadata["label_map"] = label_mapping.label_map
            state.metadata["reverse_label_map"] = label_mapping.reverse_label_map

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
        return score_judge_ranking_response(
            completion=state.output.completion,
            reverse_label_map=state.metadata.get("reverse_label_map", {}),
            alias_lookup=WORKER_MODELS,
        )

    return score


@task
def worker_measurement_adherence_eval(
    n_chunks: int = 2,
    seed: int = 42,
    workspace_id: str | None = None,
    models: str | list[str] | None = None,
    worker_timeout: int | None = None,
):
    """Evaluate worker models on measurement instruction adherence.

    A judge model ranks competing worker outputs without knowing model names.
    Returns the full ranking (e.g., "gemini > kimi > haiku") as the score answer.

    Args:
        n_chunks: Number of semantic worker chunks to evaluate
        seed: Random seed for chunk sampling
        workspace_id: Workspace to load persisted Stage 2 inputs from.
        models: Comma-separated model IDs to compete, or None for all
        worker_timeout: Timeout in seconds for each worker (default: from config, 180s)
    """
    # Parse models argument
    model_ids = parse_csv_task_arg(models)

    return Task(
        dataset=create_eval_dataset(
            n_chunks=n_chunks,
            seed=seed,
            workspace_id=workspace_id,
        ),
        solver=[
            judge_solver(model_ids=model_ids, worker_timeout=worker_timeout),
        ],
        scorer=measurement_adherence_scorer(),
    )
