"""Inspect AI evaluation for functional specification quality.

Evaluates the LLM's ability to propose a correct ModelSpec given a CausalSpec.
Tests Stage 4's propose_model_spec() against reference CausalSpecs, scoring
across 5 dimensions: likelihoods, link functions, AR structure, model clock,
and parameter constraints.

Usage:
    inspect eval evals/single_model/eval4_functional_spec.py --model anthropic/claude-sonnet-4-5-20250929
    inspect eval evals/single_model/eval4_functional_spec.py --model google/vertex/gemini-3-flash-preview
    inspect eval evals/single_model/eval4_functional_spec.py -T questions=1
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import (
    get_questions_with_causal_spec,
    select_question,
    select_questions,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from causal_ssm_agent.orchestrator.schemas_model import (
    EXPECTED_CONSTRAINT_FOR_ROLE,
    VALID_LIKELIHOODS_FOR_DTYPE,
    VALID_LINKS_FOR_DISTRIBUTION,
)
from causal_ssm_agent.orchestrator.stage4_orchestrator import propose_model_spec
from causal_ssm_agent.utils.llm import make_orchestrator_generate_fn

# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════


def build_synthetic_data_summary(causal_spec: dict) -> str:
    """Build a synthetic data summary from CausalSpec structure.

    Provides enough context for the LLM to propose a sensible model spec
    without needing real data.
    """
    indicators = causal_spec.get("measurement", {}).get("indicators", [])
    constructs = causal_spec.get("latent", {}).get("constructs", [])

    lines = ["Data Summary (synthetic from model structure):"]
    lines.append(f"  Total indicators: {len(indicators)}")
    lines.append(f"  Total constructs: {len(constructs)}")

    # Summarize dtypes
    dtypes = Counter(ind.get("measurement_dtype", "unknown") for ind in indicators)
    lines.append(f"  Data types: {dict(dtypes)}")

    # Per-indicator summary
    lines.append("  Per indicator:")
    for ind in indicators:
        name = ind.get("name", "?")
        dtype = ind.get("measurement_dtype", "?")
        construct = ind.get("construct_name", "?")
        lines.append(f"    {name}: dtype={dtype}, construct={construct}")

    # Construct temporal info
    tv_constructs = [
        c["name"]
        for c in constructs
        if c.get("temporal_status") == "time_varying" and c.get("role") == "endogenous"
    ]
    if tv_constructs:
        lines.append(f"  Time-varying endogenous constructs ({len(tv_constructs)}):")
        for name in tv_constructs[:10]:
            lines.append(f"    - {name}")
        if len(tv_constructs) > 10:
            lines.append(f"    ... and {len(tv_constructs) - 10} more")

    return "\n".join(lines)


def create_eval_dataset(questions: str | None = None) -> MemoryDataset:
    """Create evaluation dataset from available CausalSpecs.

    Args:
        questions: Optional comma-separated question selectors to filter.
    """
    available = get_questions_with_causal_spec()
    if questions:
        available = select_questions(available, questions)

    samples = []
    for q in available:
        causal_spec = q.load_causal_spec()
        data_summary = build_synthetic_data_summary(causal_spec)

        samples.append(
            Sample(
                input=f"Propose a model specification for: {q.question}",
                id=f"q_{q.slug}",
                metadata={
                    "question_slug": q.slug,
                    "question": q.question,
                    "causal_spec": causal_spec,
                    "data_summary": data_summary,
                },
            )
        )

    return MemoryDataset(samples)


# ══════════════════════════════════════════════════════════════════════════════
# Solver
# ══════════════════════════════════════════════════════════════════════════════


def functional_spec_solver():
    """Solver that runs the full Stage 4 flow using core logic.

    Calls propose_model_spec() directly (same as production). Falls back to
    raw dict scoring if Pydantic validation fails despite the feedback loop.
    """

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            model = get_model()
            gen_fn = make_orchestrator_generate_fn(model)

            causal_spec = state.metadata["causal_spec"]
            data_summary = state.metadata["data_summary"]
            question = state.metadata["question"]

            # Run the SAME core logic as production
            result = await propose_model_spec(
                causal_spec=causal_spec,
                data_summary=data_summary,
                question=question,
                generate=gen_fn,
            )
            model_spec_dict = json.loads(result.model_spec.model_dump_json())
            state.metadata["model_spec"] = model_spec_dict
            state.output.completion = result.raw_response

            # Persist for downstream evals
            question_slug = state.metadata["question_slug"]
            available = get_questions_with_causal_spec()
            q = select_question(available, question_slug)
            q.save_model_spec(model_spec_dict)

            return state

        return solve

    return _solver()


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════


def _words_match(concept_name: str, param_name: str) -> bool:
    """Check if a concept name matches a parameter name, handling abbreviations.

    Uses prefix matching: a construct word "cognitive" matches param word "cog"
    (prefix), and vice versa. Requires at least half of the construct words to
    match some param word.
    """
    construct_words = concept_name.lower().split()
    param_words = param_name.lower().replace("_", " ").split()
    # Remove common prefixes that aren't part of the construct name
    skip = {"ar", "beta", "sigma", "rho", "sd", "loading"}
    param_words = [w for w in param_words if w not in skip]
    if not construct_words:
        return False

    matched = 0
    for cw in construct_words:
        for pw in param_words:
            if cw.startswith(pw) or pw.startswith(cw):
                matched += 1
                break

    # Require at least half the construct words to match
    return matched >= len(construct_words) / 2


def _score_functional_spec(model_spec: dict, causal_spec: dict) -> dict:
    """Score a ModelSpec against a reference CausalSpec.

    Returns dict with:
        - total_points: raw points earned
        - max_points: maximum possible points
        - dimension_scores: per-dimension breakdown
    """
    indicators = causal_spec.get("measurement", {}).get("indicators", [])
    constructs = causal_spec.get("latent", {}).get("constructs", [])
    edges = causal_spec.get("latent", {}).get("edges", [])

    likelihoods = model_spec.get("likelihoods", [])
    parameters = model_spec.get("parameters", [])

    # Likelihood lookup: variable -> (distribution, link)
    likelihood_by_var = {}
    for lik in likelihoods:
        var = lik.get("variable", "")
        likelihood_by_var[var] = {
            "distribution": lik.get("distribution", ""),
            "link": lik.get("link", ""),
        }

    # ── Dimension 1: Likelihood correctness (+2 per correct) ──
    likelihood_points = 0
    likelihood_max = len(indicators) * 2
    likelihood_details = []

    for ind in indicators:
        name = ind["name"]
        dtype = ind.get("measurement_dtype", "continuous")
        valid_dists = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype, set())

        if name in likelihood_by_var:
            dist = likelihood_by_var[name]["distribution"]
            if dist in valid_dists:
                likelihood_points += 2
                likelihood_details.append(f"{name}: OK ({dist} for {dtype})")
            else:
                likelihood_details.append(
                    f"{name}: WRONG ({dist} for {dtype}, expected one of {valid_dists})"
                )
        else:
            likelihood_details.append(f"{name}: MISSING (no likelihood specified)")

    # ── Dimension 2: Link function correctness (+1 per correct, requires correct dist) ──
    link_points = 0
    link_max = len(indicators)
    link_details = []

    for ind in indicators:
        name = ind["name"]
        dtype = ind.get("measurement_dtype", "continuous")
        valid_dists = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype, set())

        if name in likelihood_by_var:
            dist = likelihood_by_var[name]["distribution"]
            link = likelihood_by_var[name]["link"]

            # Only score link if distribution is correct
            if dist in valid_dists:
                valid_links = VALID_LINKS_FOR_DISTRIBUTION.get(dist, set())
                if link in valid_links:
                    link_points += 1
                    link_details.append(f"{name}: OK ({link} for {dist})")
                else:
                    link_details.append(
                        f"{name}: WRONG ({link} for {dist}, expected one of {valid_links})"
                    )
            else:
                link_details.append(f"{name}: SKIP (distribution wrong)")
        else:
            link_details.append(f"{name}: MISSING")

    # ── Dimension 3: AR(1) structure (+3 per TV endogenous with ar_coefficient) ──
    tv_endogenous = [
        c["name"]
        for c in constructs
        if c.get("temporal_status") == "time_varying" and c.get("role") == "endogenous"
    ]
    ar_max = len(tv_endogenous) * 3

    # Check which constructs have an ar_coefficient parameter
    param_roles = {}
    for p in parameters:
        role = p.get("role", "")
        name = p.get("name", "")
        param_roles.setdefault(role, []).append(name)

    ar_params = set(param_roles.get("ar_coefficient", []))

    ar_points = 0
    ar_details = []
    for construct_name in tv_endogenous:
        # Check if any ar_coefficient parameter references this construct
        # Heuristic: substring match OR prefix-based word matching (handles abbreviations)
        construct_lower = construct_name.lower().replace(" ", "_")
        has_ar = any(
            construct_lower in p_name.lower()
            or construct_name.lower() in p_name.lower()
            or _words_match(construct_name, p_name)
            for p_name in ar_params
        )
        if has_ar:
            ar_points += 3
            ar_details.append(f"{construct_name}: OK (has AR coefficient)")
        else:
            ar_details.append(f"{construct_name}: MISSING (no AR coefficient found)")

    # ── Dimension 4: Parameter constraints (+1 per correct) ──
    constraint_points = 0
    constraint_max = len(parameters)
    constraint_details = []

    for p in parameters:
        role = p.get("role", "")
        constraint = p.get("constraint", "")
        expected = EXPECTED_CONSTRAINT_FOR_ROLE.get(role)

        if expected is not None:
            if constraint == expected:
                constraint_points += 1
                constraint_details.append(f"{p['name']}: OK ({constraint})")
            else:
                constraint_details.append(f"{p['name']}: WRONG ({constraint}, expected {expected})")
        else:
            # Unknown role — give point if any constraint is set
            constraint_points += 1
            constraint_details.append(f"{p['name']}: OK (unknown role, accepted)")

    # ── Bonuses ──
    bonus_points = 0
    bonus_max = 8  # 5 + 3
    bonus_details = []

    # Full likelihood coverage: all indicators have a likelihood
    covered_indicators = set(likelihood_by_var.keys())
    all_indicator_names = {ind["name"] for ind in indicators}
    if all_indicator_names <= covered_indicators:
        bonus_points += 5
        bonus_details.append("Full likelihood coverage: +5")
    else:
        missing = all_indicator_names - covered_indicators
        bonus_details.append(f"Missing likelihood coverage for: {missing}")

    # Full edge coverage: all edges have at least one parameter
    edge_pairs = {(e["cause"], e["effect"]) for e in edges}
    param_names_lower = {p["name"].lower() for p in parameters}
    covered_edges = 0
    for cause, effect in edge_pairs:
        cause_lower = cause.lower().replace(" ", "_")
        effect_lower = effect.lower().replace(" ", "_")
        if any(
            (cause_lower in pn and effect_lower in pn)
            or (_words_match(cause, pn) and _words_match(effect, pn))
            for pn in param_names_lower
        ):
            covered_edges += 1
    if covered_edges == len(edge_pairs) and len(edge_pairs) > 0:
        bonus_points += 3
        bonus_details.append("Full edge coverage: +3")
    else:
        bonus_details.append(f"Edge coverage: {covered_edges}/{len(edge_pairs)}")

    total_points = likelihood_points + link_points + ar_points + constraint_points + bonus_points
    max_points = likelihood_max + link_max + ar_max + constraint_max + bonus_max

    return {
        "total_points": total_points,
        "max_points": max_points,
        "score": total_points / max_points if max_points > 0 else 0.0,
        "dimensions": {
            "likelihood": {
                "points": likelihood_points,
                "max": likelihood_max,
                "details": likelihood_details,
            },
            "link": {
                "points": link_points,
                "max": link_max,
                "details": link_details,
            },
            "ar_structure": {
                "points": ar_points,
                "max": ar_max,
                "details": ar_details,
            },
            "constraints": {
                "points": constraint_points,
                "max": constraint_max,
                "details": constraint_details,
            },
            "bonuses": {
                "points": bonus_points,
                "max": bonus_max,
                "details": bonus_details,
            },
        },
    }


@scorer(metrics=[accuracy(), stderr()])
def functional_spec_scorer():
    """Score functional specification quality.

    Normalizes the multi-dimensional score to [0, 1] for the accuracy metric.
    Threshold: >= 0.5 is scored as "C" (correct).
    """

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        model_spec = state.metadata.get("model_spec")
        causal_spec = state.metadata.get("causal_spec")

        if model_spec is None:
            return Score(
                value="I",
                answer="NO_SPEC",
                explanation="No model_spec produced",
            )

        result = _score_functional_spec(model_spec, causal_spec)
        normalized = result["score"]

        # Build explanation from dimension scores
        explanation_lines = [
            f"Score: {result['total_points']}/{result['max_points']} = {normalized:.2%}",
        ]
        for dim_name, dim_data in result["dimensions"].items():
            if isinstance(dim_data.get("details"), list):
                explanation_lines.append(f"  {dim_name}: {dim_data['points']}/{dim_data['max']}")
            elif "detail" in dim_data:
                explanation_lines.append(
                    f"  {dim_name}: {dim_data['points']}/{dim_data['max']} ({dim_data['detail']})"
                )

        value = "C" if normalized >= 0.5 else "I"

        return Score(
            value=value,
            answer=f"{normalized:.2%}",
            explanation="\n".join(explanation_lines),
            metadata={
                "normalized_score": normalized,
                "total_points": result["total_points"],
                "max_points": result["max_points"],
                "dimensions": {
                    k: {"points": v["points"], "max": v["max"]}
                    for k, v in result["dimensions"].items()
                },
            },
        )

    return score


# ══════════════════════════════════════════════════════════════════════════════
# Task
# ══════════════════════════════════════════════════════════════════════════════


@task
def functional_spec_eval(questions: str | None = None):
    """Evaluate LLM's functional specification quality.

    Tests Stage 4's propose_model_spec() against available reference CausalSpecs,
    scoring on likelihoods, links, AR structure, model clock, and constraints.

    Args:
        questions: Optional comma-separated question selectors (e.g. "1,3")
    """
    return Task(
        dataset=create_eval_dataset(questions=questions),
        solver=[functional_spec_solver()],
        scorer=functional_spec_scorer(),
    )
