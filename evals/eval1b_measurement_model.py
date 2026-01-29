"""Inspect AI evaluation for Stage 1b: Measurement Model with Identifiability.

Tests the orchestrator's ability to:
1. Operationalize theoretical constructs into measurable indicators
2. Check identifiability of target causal effects
3. Request proxies for blocking confounders when needed

This evaluates data understanding, operationalization, and causal reasoning.
Requires reference latent models from Stage 1a.

Usage:
    inspect eval evals/eval1b_measurement_model.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval evals/eval1b_measurement_model.py --model openrouter/google/gemini-2.5-pro-preview-06-05
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from dataclasses import dataclass

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver, system_message

from dsem_agent.orchestrator.prompts import (
    MEASUREMENT_MODEL_REVIEW,
    MEASUREMENT_MODEL_SYSTEM,
    MEASUREMENT_MODEL_USER,
    PROXY_REQUEST_SYSTEM,
    PROXY_REQUEST_USER,
)
from dsem_agent.orchestrator.schemas import LatentModel, MeasurementModel
from dsem_agent.utils.effects import (
    get_all_treatments,
    get_outcome_from_latent_model,
)
from dsem_agent.utils.identifiability import (
    check_identifiability,
    format_identifiability_report,
)
from dsem_agent.utils.llm import get_generate_config, make_validate_measurement_model_tool, multi_turn_generate

from evals.common import (
    extract_json_from_response,
    format_chunks,
    get_eval_questions,
    get_sample_chunks_orchestrator,
    load_eval_config,
    load_latent_model_by_question_id,
)

# Load config for models
_CONFIG = load_eval_config()

# Top-tier models for orchestrator eval
MODELS = {m["id"]: m["alias"] for m in _CONFIG["orchestrator_models"]}


@dataclass
class EvalQuestion:
    """An evaluation question with metadata."""

    id: int
    question: str


def load_questions() -> list[EvalQuestion]:
    """Load evaluation questions from config."""
    return [
        EvalQuestion(id=q["id"], question=q["question"])
        for q in get_eval_questions()
    ]


def create_eval_dataset(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset by combining questions with latent models and data.

    Args:
        n_chunks: Number of chunks to sample per question
        seed: Random seed for reproducible chunk sampling
        input_file: Specific input file name, or None for latest

    Returns:
        MemoryDataset with samples for each question
    """
    questions = load_questions()

    # Sample chunks (same for all questions for fair comparison)
    chunks = get_sample_chunks_orchestrator(n_chunks, seed, input_file)
    formatted_chunks = format_chunks(chunks)

    samples = []
    for q in questions:
        # Load reference latent model for this question
        latent_model = load_latent_model_by_question_id(q.id)
        latent_json = json.dumps(latent_model, indent=2)

        # Get outcome and potential treatments from latent model
        outcome = get_outcome_from_latent_model(latent_model)
        treatments = get_all_treatments(latent_model)

        # Build the user prompt
        user_prompt = MEASUREMENT_MODEL_USER.format(
            question=q.question,
            latent_model_json=latent_json,
            dataset_summary="Personal activity data export",
            chunks=formatted_chunks,
        )

        samples.append(
            Sample(
                input=user_prompt,
                id=f"q{q.id}",
                metadata={
                    "question": q.question,
                    "latent_model": latent_model,
                    "latent_json": latent_json,
                    "outcome": outcome,
                    "treatments": treatments,
                    "chunks": chunks,  # Raw chunks for proxy request
                    "formatted_chunks": formatted_chunks,
                    "n_chunks": n_chunks,
                    "seed": seed,
                },
            )
        )

    return MemoryDataset(samples)


def _score_measurement_model(
    measurement: MeasurementModel,
    latent: LatentModel,
    initial_id_result: dict | None = None,
    final_id_result: dict | None = None,
) -> dict:
    """Score a measurement model against its latent model.

    Scoring rules:
    - +2 per valid indicator (references known construct)
    - +1 for valid dtype
    - +1 for valid aggregation
    - +1 for specific how_to_measure (>50 chars)
    - +2 bonus for multiple indicators per construct (reliability)
    - Identifiability bonuses:
      - +10 if ALL treatments identifiable from start
      - +15 if ALL treatments identifiable after proxy fix (harder!)
      - +5 if some improvement from proxy request

    Returns dict with 'total', 'indicators', and 'breakdown'.
    """
    breakdown = []
    indicator_points = {}
    total = 0.0

    construct_names = {c.name for c in latent.constructs}
    indicators_per_construct: dict[str, int] = {}

    for indicator in measurement.indicators:
        pts = 0
        details = []

        # Valid construct reference
        if indicator.construct_name in construct_names:
            pts += 2
            details.append(f"+2 references valid construct '{indicator.construct_name}'")
            indicators_per_construct[indicator.construct_name] = (
                indicators_per_construct.get(indicator.construct_name, 0) + 1
            )
        else:
            details.append(f"+0 unknown construct '{indicator.construct_name}'")

        # Valid dtype
        valid_dtypes = {"continuous", "binary", "count", "ordinal", "categorical"}
        if indicator.measurement_dtype in valid_dtypes:
            pts += 1
            details.append(f"+1 valid dtype '{indicator.measurement_dtype}'")

        # Valid aggregation (already validated by schema, but count it)
        pts += 1
        details.append(f"+1 valid aggregation '{indicator.aggregation}'")

        # Specific how_to_measure
        if len(indicator.how_to_measure) > 50:
            pts += 1
            details.append("+1 specific how_to_measure (>50 chars)")
        else:
            details.append("+0 vague how_to_measure (<50 chars)")

        indicator_points[indicator.name] = {"points": pts, "details": details}
        total += pts

    # Bonus for multiple indicators per construct (reliability)
    multi_indicator_bonus = 0
    for construct, count in indicators_per_construct.items():
        if count > 1:
            bonus = (count - 1) * 2
            multi_indicator_bonus += bonus
            breakdown.append(
                f"+{bonus} multi-indicator bonus for '{construct}' ({count} indicators)"
            )

    total += multi_indicator_bonus

    # Identifiability bonuses
    if final_id_result:
        final_non_id = len(final_id_result["non_identifiable_treatments"])
        final_id = len(final_id_result["identifiable_treatments"])

        if final_non_id == 0:
            # All identifiable in final result
            if initial_id_result and len(initial_id_result["non_identifiable_treatments"]) > 0:
                # Fixed via proxy request - harder, more points!
                breakdown.append("+15 ALL treatments identifiable after proxy fix!")
                total += 15
            else:
                # Already identifiable from start
                breakdown.append("+10 ALL treatments identifiable from start!")
                total += 10
        elif initial_id_result:
            # Check if there was improvement
            initial_non_id = len(initial_id_result["non_identifiable_treatments"])
            if final_non_id < initial_non_id:
                improved = initial_non_id - final_non_id
                breakdown.append(
                    f"+5 Improved identifiability: {improved} treatments fixed via proxies"
                )
                total += 5
            else:
                breakdown.append(
                    f"+0 No identifiability improvement ({final_non_id} still blocked)"
                )
        else:
            breakdown.append(f"+0 {final_non_id} treatments not identifiable")

    # Build breakdown summary
    breakdown.insert(0, f"INDICATORS ({len(measurement.indicators)}):")
    for name, info in indicator_points.items():
        breakdown.append(f"  {name}: {info['points']} pts")
        for d in info["details"]:
            breakdown.append(f"    {d}")

    breakdown.append(f"\nTOTAL: {total} points")

    return {
        "total": total,
        "indicators": indicator_points,
        "breakdown": "\n".join(breakdown),
        "indicators_per_construct": indicators_per_construct,
    }


async def _request_proxies(
    model,
    config,
    question: str,
    latent_model: dict,
    measurement_dict: dict,
    id_result: dict,
    chunks: list[str],
) -> dict | None:
    """Request proxy measurements for blocking confounders.

    Returns parsed proxy response or None if request fails.
    """
    # Get unique confounders
    all_confounders = set()
    for blockers in id_result["blocking_confounders"].values():
        all_confounders.update(blockers)

    # Filter to actual constructs
    construct_names = {c["name"] for c in latent_model["constructs"]}
    confounders_to_fix = [c for c in all_confounders if c in construct_names]

    if not confounders_to_fix:
        return None

    # Format blocking info
    blocking_info = "\n".join(
        [
            f"- {treatment}: blocked by {', '.join(id_result['blocking_confounders'][treatment])}"
            for treatment in sorted(id_result["non_identifiable_treatments"])
            if treatment in id_result["blocking_confounders"]
        ]
    )

    # Format data sample
    data_sample = "\n".join(chunks[:5])

    # Build proxy request prompt
    proxy_prompt = PROXY_REQUEST_USER.format(
        blocking_info=blocking_info,
        confounders_to_operationalize=", ".join(confounders_to_fix),
        latent_model_json=json.dumps(latent_model, indent=2),
        current_measurements_json=json.dumps(measurement_dict, indent=2),
        data_sample=data_sample,
    )

    messages = [
        ChatMessageSystem(content=PROXY_REQUEST_SYSTEM),
        ChatMessageUser(content=proxy_prompt),
    ]

    try:
        response = await model.generate(messages, config=config)
        json_str = extract_json_from_response(response.completion)
        if json_str:
            return json.loads(json_str)
    except Exception:
        pass

    return None


def _merge_proxies(measurement_dict: dict, proxy_response: dict) -> dict:
    """Merge proxy indicators into measurement model."""
    if not proxy_response or not proxy_response.get("new_proxies"):
        return measurement_dict

    # Copy to avoid mutating original
    result = {
        "indicators": list(measurement_dict.get("indicators", []))
    }

    for proxy in proxy_response["new_proxies"]:
        for indicator_name in proxy.get("indicators", []):
            result["indicators"].append(
                {
                    "name": indicator_name,
                    "construct": proxy["construct"],
                    "how_to_measure": f"Proxy for {proxy['construct']}: {proxy.get('justification', '')}",
                }
            )

    return result


@scorer(metrics=[mean(), stderr()])
def measurement_model_scorer():
    """Score measurement model proposals with full identifiability flow.

    Scores the final measurement model after the complete Stage 1b flow:
    1. Initial proposal
    2. Identifiability check
    3. Proxy request if needed
    4. Final identifiability check
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        latent_data = state.metadata.get("latent_model", {})

        # Parse latent model from metadata
        try:
            latent = LatentModel(**latent_data)
        except Exception as e:
            return Score(
                value=0.0,
                answer="[Invalid latent model in metadata]",
                explanation=f"ERROR: Could not parse latent model - {e}",
            )

        # Extract JSON from response
        json_str = extract_json_from_response(completion)
        if json_str is None:
            return Score(
                value=0.0,
                answer="[No valid JSON found]",
                explanation=(
                    "ERROR: Could not extract JSON from model response.\n"
                    f"Response preview: {completion[:500]}..."
                ),
            )

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return Score(
                value=0.0,
                answer=json_str[:200] + "..." if len(json_str) > 200 else json_str,
                explanation=f"ERROR: JSON parse failed - {e}",
            )

        # Validate against schema
        try:
            measurement = MeasurementModel(**data)
        except Exception as e:
            return Score(
                value=0.0,
                answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
                explanation=f"ERROR: Schema validation failed - {e}",
            )

        # Get identifiability results from state metadata (set by solver)
        initial_id_result = state.metadata.get("initial_identifiability")
        final_id_result = state.metadata.get("final_identifiability")

        # Score the measurement model
        scoring = _score_measurement_model(
            measurement, latent, initial_id_result, final_id_result
        )

        return Score(
            value=scoring["total"],
            answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
            explanation=scoring["breakdown"],
            metadata={
                "indicators": scoring["indicators"],
                "n_indicators": len(measurement.indicators),
                "indicators_per_construct": scoring["indicators_per_construct"],
                "initial_identifiability": initial_id_result,
                "final_identifiability": final_id_result,
                "proxy_requested": state.metadata.get("proxy_requested", False),
            },
        )

    return score


def measurement_model_solver():
    """Custom solver implementing the full Stage 1b flow.

    1. Initial measurement proposal with self-review
    2. Check identifiability
    3. If non-identifiable treatments, request proxies
    4. Merge proxies and re-check identifiability
    """

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            model = get_model()
            config = get_generate_config()

            # Get metadata
            latent_data = state.metadata.get("latent_model", {})
            latent = LatentModel(**latent_data)
            question = state.metadata.get("question", "")
            chunks = state.metadata.get("chunks", [])

            # Create validation tool bound to this sample's latent model
            tool = make_validate_measurement_model_tool(latent)

            # Step 1: Initial measurement proposal with review
            completion = await multi_turn_generate(
                messages=list(state.messages),
                model=model,
                follow_ups=[MEASUREMENT_MODEL_REVIEW],
                tools=[tool],
                config=config,
            )

            # Parse the initial measurement model
            json_str = extract_json_from_response(completion)
            if not json_str:
                state.output.completion = completion
                return state

            try:
                measurement_dict = json.loads(json_str)
            except json.JSONDecodeError:
                state.output.completion = completion
                return state

            # Step 2: Check identifiability
            latent_dict = latent.model_dump()
            initial_id_result = check_identifiability(latent_dict, measurement_dict)
            state.metadata["initial_identifiability"] = initial_id_result

            # Step 3: If non-identifiable, request proxies
            if initial_id_result["non_identifiable_treatments"]:
                state.metadata["proxy_requested"] = True

                proxy_response = await _request_proxies(
                    model,
                    config,
                    question,
                    latent_dict,
                    measurement_dict,
                    initial_id_result,
                    chunks,
                )

                if proxy_response and proxy_response.get("new_proxies"):
                    # Merge proxies
                    measurement_dict = _merge_proxies(measurement_dict, proxy_response)

                    # Re-check identifiability
                    final_id_result = check_identifiability(latent_dict, measurement_dict)
                    state.metadata["final_identifiability"] = final_id_result

                    # Update completion with merged measurement
                    completion = json.dumps(measurement_dict, indent=2)
                else:
                    # No proxies found, final = initial
                    state.metadata["final_identifiability"] = initial_id_result
            else:
                # Already identifiable
                state.metadata["proxy_requested"] = False
                state.metadata["final_identifiability"] = initial_id_result

            state.output.completion = completion
            return state

        return solve

    return _solver()


@task
def measurement_model_eval(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
):
    """Evaluate LLM ability to operationalize constructs with identifiability.

    Stage 1b evaluation with full identifiability flow:
    1. Initial proposal from latent model + data
    2. Self-review focusing on operationalization coherence
    3. Identifiability check
    4. Proxy request for blocking confounders (if needed)
    5. Re-check identifiability after merging proxies

    Scoring includes bonuses for:
    - All effects identifiable from start (+10)
    - All effects identifiable after proxy fix (+15)
    - Partial improvement from proxy request (+5)

    Args:
        n_chunks: Number of data chunks to include in each sample
        seed: Random seed for chunk sampling (reproducibility)
        input_file: Specific preprocessed file name, or None for latest
    """
    return Task(
        dataset=create_eval_dataset(n_chunks=n_chunks, seed=seed, input_file=input_file),
        solver=[
            system_message(MEASUREMENT_MODEL_SYSTEM),
            measurement_model_solver(),
        ],
        scorer=measurement_model_scorer(),
    )
