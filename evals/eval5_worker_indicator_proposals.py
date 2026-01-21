"""Inspect AI evaluation for worker indicator proposal quality.

Uses a judge model (from config.yaml) to evaluate the relevance of indicators
proposed by a single worker model. The worker is scored by acceptance rate -
how many of its proposed indicators the judge accepts as genuinely relevant.

Usage:
    inspect eval evals/eval5_worker_indicator_proposals.py
"""

import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from causal_agent.workers.prompts import WORKER_W_PROPOSALS_SYSTEM, WORKER_USER
from causal_agent.workers.agents import (
    _format_indicators,
    _get_outcome_description,
)
from causal_agent.utils.llm import get_generate_config, make_worker_tools, multi_turn_generate, parse_json_response

from evals.common import (
    get_eval_questions,
    get_sample_chunks_worker,
    load_eval_config,
    load_dsem_model_by_question_id,
)


# Load config
_CONFIG = load_eval_config()

# Default worker model for indicator proposals (gemini 3 flash - best at extraction)
DEFAULT_WORKER_MODEL = "google/vertex/gemini-3-flash-preview"

# Judge model from config
JUDGE_MODEL = _CONFIG.get("judge_model", "openrouter/anthropic/claude-sonnet-4")


JUDGE_SYSTEM = """\
You are an expert evaluator assessing proposed causal indicators. A worker model was given:
1. A causal research question
2. An existing schema of indicators (variables) to extract
3. A data chunk

The worker may propose new indicators it believes are causally relevant but missing from the schema. Your task is to evaluate each proposed indicator and decide if it should be ACCEPTED or REJECTED.

## Acceptance Criteria

A proposed indicator should be ACCEPTED if ALL of the following are true:
1. **Causally relevant**: It has a plausible causal relationship to the outcome or other constructs in the question
2. **Not redundant**: It's genuinely distinct from existing indicators (not just a rewording or subset)
3. **Measurable**: It can actually be extracted from the type of data being processed
4. **Well-evidenced**: The worker provides concrete evidence from the chunk showing the indicator exists

A proposed indicator should be REJECTED if ANY of the following are true:
1. It's only tangentially related to the causal question
2. It duplicates or overlaps substantially with an existing indicator
3. It's too vague or abstract to measure
4. The evidence provided doesn't support its existence in the data
5. It's based on speculation rather than observation

## Output Format

```json
{
  "evaluations": [
    {
      "indicator_name": "proposed_indicator_name",
      "verdict": "ACCEPTED" | "REJECTED",
      "rationale": "Brief explanation of your decision"
    }
  ],
  "accepted_count": 2,
  "total_count": 3
}
```

If the worker proposed no new indicators, return:
```json
{
  "evaluations": [],
  "accepted_count": 0,
  "total_count": 0
}
```
"""

JUDGE_USER = """\
## Causal Question

{question}

## Existing Schema Indicators

The worker was given this schema to extract from. These indicators already exist:

{existing_indicators}

## Outcome Variable

{outcome_description}

## Data Chunk (what the worker saw)

{chunk}

## Proposed Indicators

{proposals}

Please evaluate each proposed indicator and decide if it should be ACCEPTED or REJECTED.
"""


async def generate_worker_output(
    model_id: str,
    chunk: str,
    question: str,
    dsem_model: dict,
) -> str:
    """Generate worker output for the model.

    Returns the raw completion text (including JSON).
    """
    model = get_model(model_id)

    indicators_text = _format_indicators(dsem_model)
    outcome_description = _get_outcome_description(dsem_model)

    messages = [
        ChatMessageSystem(content=WORKER_W_PROPOSALS_SYSTEM),
        ChatMessageUser(
            content=WORKER_USER.format(
                question=question,
                outcome_description=outcome_description,
                indicators=indicators_text,
                chunk=chunk,
            )
        ),
    ]

    config = get_generate_config()

    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        tools=make_worker_tools(dsem_model),
        config=config,
    )

    return completion


def extract_proposed_indicators(output: str) -> list[dict]:
    """Extract proposed indicators from worker output.

    Args:
        output: Raw completion text from worker

    Returns:
        List of proposed indicator dicts, or empty list if none/error
    """
    try:
        data = parse_json_response(output)
        proposed = data.get("proposed_indicators")
        if proposed and isinstance(proposed, list):
            return proposed
        return []
    except Exception:
        return []


def format_proposals_for_judge(proposals: list[dict]) -> str:
    """Format proposed indicators for the judge prompt."""
    if not proposals:
        return "[No new indicators proposed]"
    return json.dumps(proposals, indent=2)


def format_existing_indicators(dsem_model: dict) -> str:
    """Format existing indicators from DSEMModel for judge context."""
    indicators = dsem_model.get("measurement", {}).get("indicators", [])
    lines = []
    for ind in indicators:
        name = ind.get("name", "unknown")
        construct_name = ind.get("construct") or ind.get("construct_name", "")
        how = ind.get("how_to_measure", "")
        construct_info = f" (measures: {construct_name})" if construct_name else ""
        lines.append(f"- **{name}**{construct_info}: {how}")
    return "\n".join(lines)


def create_eval_dataset(
    n_chunks: int = 3,
    seed: int = 55,  # Different seed than other evals (42) for unique chunks
    input_file: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset.

    Each sample contains:
    - A question from the eval set with its corresponding DSEMModel
    - A data chunk
    - Metadata with schema for judge evaluation

    Cycles through question-DSEMModel pairs to ensure coverage.

    Args:
        n_chunks: Number of chunks per question
        seed: Random seed for reproducibility
        input_file: Specific input file name, or None for latest

    Returns:
        MemoryDataset with samples
    """
    # Get all questions
    questions = get_eval_questions()

    # Get chunks - total needed across all questions
    total_chunks = n_chunks * len(questions)
    chunks = get_sample_chunks_worker(total_chunks, seed, input_file)

    samples = []
    chunk_idx = 0

    for q in questions:
        # Load the DSEMModel for this specific question
        dsem_model = load_dsem_model_by_question_id(q["id"])
        indicators_text = _format_indicators(dsem_model)
        outcome_description = _get_outcome_description(dsem_model)
        existing_inds_text = format_existing_indicators(dsem_model)

        for i in range(n_chunks):
            if chunk_idx >= len(chunks):
                break

            chunk = chunks[chunk_idx]
            chunk_idx += 1

            samples.append(
                Sample(
                    input=f"Question: {q['question']}\nChunk index: {i}",
                    id=f"q{q['id']}_chunk{i}",
                    metadata={
                        "question_id": q["id"],
                        "question": q["question"],
                        "chunk": chunk,
                        "chunk_index": i,
                        "dsem_model": dsem_model,
                        "existing_indicators": existing_inds_text,
                        "outcome_description": outcome_description,
                    },
                )
            )

    return MemoryDataset(samples)


def judge_solver(worker_model: str | None = None, worker_timeout: float | None = None):
    """Solver that generates worker output and asks judge to evaluate proposals.

    Args:
        worker_model: Model ID for the worker. If None, uses default (gemini 3 flash).
        worker_timeout: Timeout in seconds for the worker. If None, uses config default.
    """
    if worker_model is None:
        worker_model = DEFAULT_WORKER_MODEL

    # Get timeout from config if not specified
    if worker_timeout is None:
        worker_timeout = _CONFIG.get("worker_timeout_seconds", 180)

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            dsem_model = state.metadata["dsem_model"]
            question = state.metadata["question"]
            chunk = state.metadata["chunk"]
            existing_inds = state.metadata["existing_indicators"]
            outcome_description = state.metadata["outcome_description"]

            # Generate worker output
            try:
                worker_output = await asyncio.wait_for(
                    generate_worker_output(worker_model, chunk, question, dsem_model),
                    timeout=worker_timeout,
                )
            except asyncio.TimeoutError:
                worker_output = f"[TIMEOUT: Worker did not finish within {worker_timeout}s]"
            except Exception as e:
                worker_output = f"[ERROR: {e}]"

            # Extract proposed indicators
            proposals = extract_proposed_indicators(worker_output)

            # Store in metadata for scorer
            state.metadata["worker_output"] = worker_output
            state.metadata["proposals"] = proposals

            # Format proposals for judge
            proposals_text = format_proposals_for_judge(proposals)

            # Build judge prompt
            judge_prompt = JUDGE_USER.format(
                question=question,
                existing_indicators=existing_inds,
                outcome_description=outcome_description,
                chunk=chunk[:3000] + "..." if len(chunk) > 3000 else chunk,
                proposals=proposals_text,
            )

            # Replace messages with judge prompt
            state.messages = [
                ChatMessageSystem(content=JUDGE_SYSTEM),
                ChatMessageUser(content=judge_prompt),
            ]

            # Generate judge response
            judge = get_model(JUDGE_MODEL)
            response = await judge.generate(state.messages, config=get_generate_config())
            state.output.completion = response.completion

            return state

        return solve

    return _solver()


@scorer(metrics=[mean(), stderr()])
def indicator_proposal_scorer():
    """Score based on acceptance rate of proposed indicators.

    Returns:
        - Score value is acceptance rate (accepted / total), or 1.0 if no proposals
        - Answer shows "X/Y accepted"
        - Explanation contains judge rationales
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        proposals = state.metadata.get("proposals", [])

        # If no proposals, score as 1.0 (not penalized for not proposing)
        if not proposals:
            return Score(
                value=1.0,
                answer="No proposals (0/0)",
                explanation="Worker did not propose any new indicators",
                metadata={"accepted": 0, "total": 0, "evaluations": []},
            )

        # Extract JSON from judge response
        try:
            import re
            json_match = re.search(r"\{[\s\S]*\}", completion)
            if not json_match:
                return Score(
                    value=0.0,
                    answer="[No JSON found in judge response]",
                    explanation=f"Judge response: {completion[:500]}...",
                )

            judge_data = json.loads(json_match.group())
            evaluations = judge_data.get("evaluations", [])
            accepted_count = judge_data.get("accepted_count", 0)
            total_count = judge_data.get("total_count", len(proposals))

        except json.JSONDecodeError as e:
            return Score(
                value=0.0,
                answer="[JSON parse error]",
                explanation=f"Error: {e}\nResponse: {completion[:500]}...",
            )

        # Calculate acceptance rate
        if total_count > 0:
            acceptance_rate = accepted_count / total_count
        else:
            acceptance_rate = 1.0

        # Build explanation from evaluations
        explanation_parts = []
        for ev in evaluations:
            ind_name = ev.get("indicator_name") or ev.get("dimension_name", "?")
            verdict = ev.get("verdict", "?")
            rationale = ev.get("rationale", "")
            explanation_parts.append(f"{ind_name}: {verdict} - {rationale}")

        explanation = "\n".join(explanation_parts) if explanation_parts else "No evaluations"

        return Score(
            value=acceptance_rate,
            answer=f"{accepted_count}/{total_count} accepted",
            explanation=explanation,
            metadata={
                "accepted": accepted_count,
                "total": total_count,
                "evaluations": evaluations,
            },
        )

    return score


@task
def worker_indicator_proposals_eval(
    n_chunks: int = 2,
    seed: int = 55,
    input_file: str | None = None,
    worker_model: str | None = None,
    worker_timeout: int | None = None,
):
    """Evaluate indicator proposal quality from a single worker model.

    A judge model evaluates the relevance of indicators proposed by the worker.
    Score is the acceptance rate of proposals.

    Args:
        n_chunks: Number of chunks per question (total samples = n_chunks * 5 questions)
        seed: Random seed for chunk sampling
        input_file: Specific preprocessed file name, or None for latest
        worker_model: Model ID for the worker (default: gemini 3 flash)
        worker_timeout: Timeout in seconds for the worker (default: from config)
    """
    return Task(
        dataset=create_eval_dataset(n_chunks=n_chunks, seed=seed, input_file=input_file),
        solver=[
            judge_solver(worker_model=worker_model, worker_timeout=worker_timeout),
        ],
        scorer=indicator_proposal_scorer(),
    )
