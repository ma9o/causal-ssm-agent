"""Inspect AI evaluation for Stage 4: LLM-Assisted Prior Elicitation.

Tests the LLM's ability to elicit domain-informed priors for causal effect
parameters. Evaluates:

1. Format compliance (valid JSON with required fields)
2. Constraint satisfaction (AR in [0,1], variances positive)
3. Sign correctness (known-direction effects)
4. Magnitude reasonableness (plausible effect sizes)
5. Uncertainty calibration (reasonable std values)
6. Domain reasoning quality (substantive justifications)

Uses the same core logic as production (via run_stage4), ensuring the eval
tests what actually runs.

Usage:
    inspect eval evals/eval5_prior_elicitation.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval evals/eval5_prior_elicitation.py --model openrouter/google/gemini-2.5-pro-preview-06-05
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from dataclasses import dataclass

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver, system_message

from dsem_agent.flows.stages.stage4_model import specify_model, DEFAULT_PRIORS
from dsem_agent.orchestrator.prompts import PRIOR_ELICITATION_SYSTEM
from dsem_agent.orchestrator.stage4 import run_stage4, Stage4Result
from dsem_agent.utils.llm import make_orchestrator_generate_fn

from evals.common import (
    get_eval_questions,
    load_dsem_model_by_question_id,
    load_eval_config,
)


# Load config
_CONFIG = load_eval_config()
MODELS = {m["id"]: m["alias"] for m in _CONFIG["orchestrator_models"]}


# Expected effect signs based on domain knowledge
# Format: {edge_pattern: expected_sign} where sign is "negative", "positive", or None (unknown)
EXPECTED_SIGNS = {
    # Common psychological/behavioral relationships
    ("Stress", "Mood"): "negative",
    ("Stress", "Focus"): "negative",
    ("Stress", "Sleep"): "negative",
    ("Sleep", "Fatigue"): "negative",  # better sleep → less fatigue
    ("Sleep", "Cognitive"): "positive",  # better sleep → better cognition
    ("Fatigue", "Focus"): "negative",
    ("Fatigue", "Performance"): "negative",
    ("Workload", "Stress"): "positive",
    ("Workload", "Fatigue"): "positive",
    ("Deadline", "Stress"): "positive",
    ("Interruption", "Focus"): "negative",
    ("Confidence", "Performance"): "positive",
    ("Experience", "Performance"): "positive",
    ("Experience", "Confidence"): "positive",
    # Error resolution domain
    ("Error Complexity", "Error Resolution Speed"): "negative",
    ("Cognitive Fatigue", "Error Resolution Speed"): "negative",
    ("Focus Quality", "Error Resolution Speed"): "positive",
    ("Debugging Confidence", "Error Resolution Speed"): "positive",
    ("Team Support", "Error Resolution Speed"): "positive",
}


def _get_expected_sign(cause: str, effect: str) -> str | None:
    """Look up expected sign for a causal relationship."""
    # Try exact match first
    if (cause, effect) in EXPECTED_SIGNS:
        return EXPECTED_SIGNS[(cause, effect)]

    # Try partial matching (keywords)
    cause_lower = cause.lower()
    effect_lower = effect.lower()

    for (c_pattern, e_pattern), sign in EXPECTED_SIGNS.items():
        c_lower = c_pattern.lower()
        e_lower = e_pattern.lower()

        # Check if patterns are substrings
        if c_lower in cause_lower and e_lower in effect_lower:
            return sign

    return None


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


def create_eval_dataset() -> MemoryDataset:
    """Create evaluation dataset from existing DSEM models.

    Each sample contains a full DSEMModel and its associated question,
    which gets converted to a ModelSpec for prior elicitation.
    """
    questions = load_questions()

    samples = []
    for q in questions:
        dsem_model = load_dsem_model_by_question_id(q.id)

        # Pre-compute model_spec using rule-based specification
        model_spec = specify_model.fn(
            latent_dict=dsem_model.get("latent", {}),
            dsem_dict=dsem_model,
        )

        samples.append(
            Sample(
                input=q.question,
                id=f"q{q.id}",
                metadata={
                    "question": q.question,
                    "dsem_model": dsem_model,
                    "model_spec": model_spec,
                },
            )
        )

    return MemoryDataset(samples)


def _score_prior(
    param_name: str,
    prior: dict,
    edge_specs: dict,
) -> dict:
    """Score a single prior.

    Returns dict with points breakdown and details.
    """
    points = 0.0
    details = []
    issues = []

    mean_val = prior.get("mean")
    std_val = prior.get("std")
    reasoning = prior.get("reasoning", "")
    source = prior.get("source", "unknown")

    # 1. Format compliance: required fields exist and are valid
    if mean_val is not None and isinstance(mean_val, (int, float)):
        points += 1
        details.append("+1 valid mean")
    else:
        issues.append("missing/invalid mean")

    if std_val is not None and isinstance(std_val, (int, float)):
        points += 1
        details.append("+1 valid std")
    else:
        issues.append("missing/invalid std")

    # 2. Constraint satisfaction
    if param_name.startswith("rho_"):
        # AR coefficient: mean should be in [0, 1]
        if mean_val is not None and 0 <= mean_val <= 1:
            points += 1
            details.append("+1 AR mean in [0,1]")
        else:
            issues.append(f"AR mean {mean_val} outside [0,1]")

    elif param_name.startswith("sigma_"):
        # Residual variance: mean should be positive
        if mean_val is not None and mean_val > 0:
            points += 1
            details.append("+1 sigma positive")
        else:
            issues.append(f"sigma mean {mean_val} not positive")

    elif param_name.startswith("beta_"):
        # Cross-lag coefficient: magnitude check (standardized effects rarely > 2)
        if mean_val is not None and abs(mean_val) <= 3:
            points += 1
            details.append("+1 beta in plausible range")
        else:
            issues.append(f"beta mean {mean_val} seems extreme")

    elif param_name.startswith("lambda_"):
        # Factor loading: should be positive
        if mean_val is not None and mean_val > 0:
            points += 1
            details.append("+1 loading positive")
        else:
            issues.append(f"loading mean {mean_val} not positive")

    # 3. Uncertainty calibration: std should be reasonable
    if std_val is not None and 0.01 <= std_val <= 10:
        points += 1
        details.append("+1 std in reasonable range")
    elif std_val is not None:
        if std_val < 0.01:
            issues.append(f"std {std_val} overconfident")
        elif std_val > 10:
            issues.append(f"std {std_val} too uncertain")

    # 4. Reasoning quality
    if reasoning and len(reasoning) > 20:
        points += 1
        details.append("+1 substantive reasoning")

        # Bonus: domain-relevant reasoning
        domain_keywords = [
            "effect", "relationship", "research", "literature", "study",
            "evidence", "expect", "domain", "typically", "mechanism",
            "causal", "positive", "negative", "increase", "decrease",
        ]
        if any(kw in reasoning.lower() for kw in domain_keywords):
            points += 0.5
            details.append("+0.5 domain-relevant reasoning")
    else:
        issues.append("missing/brief reasoning")

    # 5. Sign correctness (for beta parameters)
    if param_name.startswith("beta_") and mean_val is not None:
        edge_spec = edge_specs.get(param_name, {})
        cause = edge_spec.get("cause", "")
        effect = edge_spec.get("effect", "")
        expected_sign = _get_expected_sign(cause, effect)

        if expected_sign is not None:
            actual_sign = "positive" if mean_val > 0 else "negative" if mean_val < 0 else "zero"
            if expected_sign == actual_sign or (expected_sign == "positive" and mean_val >= 0) or (expected_sign == "negative" and mean_val <= 0):
                points += 2
                details.append(f"+2 correct sign ({expected_sign})")
            elif mean_val == 0:
                # Zero is neutral, partial credit
                points += 0.5
                details.append(f"+0.5 zero (expected {expected_sign})")
            else:
                issues.append(f"wrong sign: got {actual_sign}, expected {expected_sign}")

    # 6. LLM vs default source
    if source == "llm":
        points += 1
        details.append("+1 LLM-elicited (not fallback)")
    elif source == "llm_aggregated":
        points += 1.5
        details.append("+1.5 LLM-aggregated")
    else:
        issues.append("fell back to default")

    return {
        "points": points,
        "details": details,
        "issues": issues,
        "max_possible": 9.5,  # Theoretical max for a beta with known sign
    }


def _score_stage4_result(
    result: Stage4Result,
    model_spec: dict,
) -> dict:
    """Score a complete Stage 4 result."""
    breakdown = []
    total = 0.0
    max_total = 0.0
    param_scores = {}

    priors_dict = result.to_prior_dict()
    edge_specs = model_spec.get("edges", {})

    # Score each parameter
    for param_name, prior in priors_dict.items():
        score_info = _score_prior(param_name, prior, edge_specs)
        param_scores[param_name] = score_info
        total += score_info["points"]
        max_total += score_info["max_possible"]

    # Build breakdown summary
    breakdown.append(f"PRIORS ({len(priors_dict)} parameters):")

    # Group by type
    ar_params = [p for p in priors_dict if p.startswith("rho_")]
    beta_params = [p for p in priors_dict if p.startswith("beta_")]
    sigma_params = [p for p in priors_dict if p.startswith("sigma_")]
    lambda_params = [p for p in priors_dict if p.startswith("lambda_")]

    for group_name, params in [
        ("AR coefficients", ar_params),
        ("Causal effects", beta_params),
        ("Residual variances", sigma_params),
        ("Factor loadings", lambda_params),
    ]:
        if params:
            breakdown.append(f"\n{group_name}:")
            for p in params[:5]:  # Limit to first 5 per group
                info = param_scores[p]
                pts = info["points"]
                issues = info["issues"]
                issue_str = f" [{', '.join(issues)}]" if issues else ""
                breakdown.append(f"  {p}: {pts:.1f} pts{issue_str}")
            if len(params) > 5:
                breakdown.append(f"  ... and {len(params) - 5} more")

    # Summary stats
    n_llm = sum(1 for p in priors_dict.values() if p.get("source") == "llm")
    n_default = sum(1 for p in priors_dict.values() if p.get("source") == "default")

    breakdown.append(f"\nSOURCES: {n_llm} LLM-elicited, {n_default} defaults")

    # Normalize to 0-100 scale for easier comparison
    normalized_score = (total / max_total * 100) if max_total > 0 else 0

    breakdown.append(f"\nTOTAL: {total:.1f} / {max_total:.1f} ({normalized_score:.1f}%)")

    return {
        "total": total,
        "max_total": max_total,
        "normalized": normalized_score,
        "breakdown": "\n".join(breakdown),
        "param_scores": param_scores,
        "n_llm": n_llm,
        "n_default": n_default,
    }


@scorer(metrics=[mean(), stderr()])
def prior_elicitation_scorer():
    """Score prior elicitation results.

    Returns normalized score (0-100) based on:
    - Format compliance
    - Constraint satisfaction
    - Sign correctness
    - Magnitude reasonableness
    - Uncertainty calibration
    - Domain reasoning quality
    """

    async def score(state: TaskState, target: Target) -> Score:
        result: Stage4Result | None = state.metadata.get("stage4_result")

        if result is None:
            return Score(
                value=0.0,
                answer="[No result]",
                explanation="Stage 4 did not produce a result",
            )

        model_spec = state.metadata.get("model_spec", {})
        scoring = _score_stage4_result(result, model_spec)

        # Build answer preview
        priors_preview = result.to_prior_dict()
        answer_lines = []
        for param, prior in list(priors_preview.items())[:5]:
            answer_lines.append(
                f"{param}: mean={prior['mean']:.2f}, std={prior['std']:.2f}"
            )
        if len(priors_preview) > 5:
            answer_lines.append(f"... and {len(priors_preview) - 5} more")

        return Score(
            value=scoring["normalized"],
            answer="\n".join(answer_lines),
            explanation=scoring["breakdown"],
            metadata={
                "raw_score": scoring["total"],
                "max_score": scoring["max_total"],
                "n_parameters": len(priors_preview),
                "n_llm_elicited": scoring["n_llm"],
                "n_defaults": scoring["n_default"],
            },
        )

    return score


def prior_elicitation_solver():
    """Solver that runs the full Stage 4 prior elicitation flow."""

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            model = get_model()
            generate_fn = make_orchestrator_generate_fn(model)

            question = state.metadata.get("question", "")
            model_spec = state.metadata.get("model_spec", {})

            # Run the SAME core logic as production
            result = await run_stage4(
                model_spec=model_spec,
                question=question,
                generate=generate_fn,
                default_priors=DEFAULT_PRIORS,
                n_paraphrases=1,
                literature_context="",  # Skip literature for eval speed
            )

            # Store result in metadata for scorer
            state.metadata["stage4_result"] = result
            state.output.completion = json.dumps(result.to_prior_dict(), indent=2)

            return state

        return solve

    return _solver()


@task
def prior_elicitation_eval():
    """Evaluate LLM ability to elicit domain-informed Bayesian priors.

    Stage 4 evaluation:
    - Input: Model specification (constructs, edges, measurement) + research question
    - Output: Prior distributions (mean, std, reasoning) for all parameters

    Scoring dimensions:
    - Format compliance and validity
    - Statistical constraints (AR in [0,1], variances positive)
    - Domain knowledge (effect signs, magnitudes)
    - Reasoning quality
    """
    return Task(
        dataset=create_eval_dataset(),
        solver=[
            system_message(PRIOR_ELICITATION_SYSTEM),
            prior_elicitation_solver(),
        ],
        scorer=prior_elicitation_scorer(),
    )
