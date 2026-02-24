"""Inspect AI evaluation for prior elicitation quality.

Evaluates the full prior elicitation pipeline: Exa literature search,
LLM prior elicitation, and prior predictive validation. Runs the exact
same production code path (search_parameter_literature -> elicit_prior ->
validate_prior_predictive) under the inspect framework.

Usage:
    inspect eval evals/single_model/eval4b_prior_elicitation.py \
      --model openrouter/anthropic/claude-haiku-4.5 \
      -T questions=1 n_params=3
"""

import asyncio
import logging
import math
import random
import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.common import (
    get_questions_with_model_spec_and_causal_spec,
    select_questions,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from causal_ssm_agent.models.prior_predictive import (
    _compute_data_stats,
    format_parameter_feedback,
    get_failed_parameters,
    validate_prior_predictive,
)
from causal_ssm_agent.orchestrator.schemas_model import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
)
from causal_ssm_agent.utils.llm import make_worker_generate_fn
from causal_ssm_agent.workers.prior_research import (
    elicit_prior,
    get_default_prior,
    search_parameter_literature,
)
from causal_ssm_agent.workers.prompts.prior_research import format_literature_for_parameter

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Parameter selection
# ══════════════════════════════════════════════════════════════════════════════

# Priority order for diverse role sampling
_ROLE_PRIORITY = [
    ParameterRole.FIXED_EFFECT,
    ParameterRole.AR_COEFFICIENT,
    ParameterRole.RESIDUAL_SD,
    ParameterRole.RANDOM_INTERCEPT_SD,
    ParameterRole.LOADING,
    ParameterRole.CORRELATION,
]


def select_diverse_parameters(
    parameters: list[ParameterSpec],
    n_params: int,
    seed: int = 42,
) -> list[ParameterSpec]:
    """Select a diverse subset of parameters across roles.

    Picks one parameter per role (in priority order), then fills remaining
    slots randomly from the rest.
    """
    rng = random.Random(seed)
    selected: list[ParameterSpec] = []
    remaining = list(parameters)

    # Pick one per role in priority order
    for role in _ROLE_PRIORITY:
        if len(selected) >= n_params:
            break
        candidates = [p for p in remaining if p.role == role]
        if candidates:
            pick = rng.choice(candidates)
            selected.append(pick)
            remaining.remove(pick)

    # Fill remaining slots randomly
    if len(selected) < n_params and remaining:
        extra = rng.sample(remaining, min(n_params - len(selected), len(remaining)))
        selected.extend(extra)

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════


def create_eval_dataset(
    questions: str | None = None,
    n_params: int | None = None,
) -> MemoryDataset:
    """Create evaluation dataset from questions with both model_spec and causal_spec.

    Args:
        questions: Optional comma-separated question selectors to filter.
        n_params: Max parameters to elicit (None = all). Remaining get defaults.
    """
    available = get_questions_with_model_spec_and_causal_spec()
    if questions:
        available = select_questions(available, questions)

    if not available:
        raise ValueError(
            "No questions found with both model_spec.json and causal_spec.json. "
            "Run eval4_functional_spec first to generate model_spec."
        )

    samples = []
    for q in available:
        model_spec = q.load_model_spec()
        causal_spec = q.load_causal_spec()

        samples.append(
            Sample(
                input=f"Elicit priors for model parameters: {q.question}",
                id=f"q_{q.slug}",
                metadata={
                    "question_slug": q.slug,
                    "question": q.question,
                    "model_spec": model_spec,
                    "causal_spec": causal_spec,
                    "n_params": n_params,
                },
            )
        )

    return MemoryDataset(samples)


# ══════════════════════════════════════════════════════════════════════════════
# Solver
# ══════════════════════════════════════════════════════════════════════════════

MAX_PRIOR_RETRIES = 3  # Matches pipeline default (config.pipeline.max_prior_retries)


def _parse_parameters(model_spec_dict: dict) -> list[ParameterSpec]:
    """Parse ParameterSpec objects from model_spec dict (strict).

    All parameter roles must be defined in the ParameterRole enum.
    """
    return [ParameterSpec.model_validate(p) for p in model_spec_dict.get("parameters", [])]


def _add_correlation_parameters(
    parameter_specs: list[dict],
    causal_spec: dict,
) -> list[dict]:
    """Auto-add correlation parameters for marginalized confounders.

    Mirrors the pipeline's stage4_orchestrated_flow logic: when an unobserved
    confounder is marginalized, its observed children have correlated innovations.
    """
    id_status = causal_spec.get("identifiability", {})
    if not id_status:
        return parameter_specs

    from causal_ssm_agent.utils.identifiability import (
        get_correlation_pairs_from_marginalization,
    )

    cor_pairs = get_correlation_pairs_from_marginalization(
        causal_spec.get("latent", {}),
        causal_spec.get("measurement", {}),
        id_status,
    )
    existing_names = {p.get("name") for p in parameter_specs}
    for s1, s2, confounder in cor_pairs:
        name = f"cor_{s1}_{s2}"
        if name not in existing_names:
            parameter_specs.append(
                {
                    "name": name,
                    "role": "correlation",
                    "constraint": "correlation",
                    "description": (
                        f"Residual correlation between {s1} and {s2} "
                        f"(marginalized confounder: {confounder})"
                    ),
                    "search_context": "",
                }
            )
            existing_names.add(name)

    return parameter_specs


def prior_elicitation_solver():
    """Solver that runs the full prior elicitation pipeline.

    Mirrors stage4_orchestrated_flow: literature search, LLM elicitation,
    prior predictive validation, and retry with feedback.
    """

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            model = get_model()
            gen_fn = make_worker_generate_fn(model)

            model_spec_dict = state.metadata["model_spec"]
            causal_spec = state.metadata["causal_spec"]
            question = state.metadata["question"]
            n_params = state.metadata.get("n_params")
            raw_data = state.metadata.get("raw_data")

            # Auto-add correlation parameters for marginalized confounders
            parameter_specs = model_spec_dict.get("parameters", [])
            parameter_specs = _add_correlation_parameters(parameter_specs, causal_spec)
            model_spec_dict["parameters"] = parameter_specs

            # Parse all parameters (strict — all roles must be in enum)
            all_parameters = _parse_parameters(model_spec_dict)

            # Select subset if n_params is set
            if n_params is not None and n_params < len(all_parameters):
                elicit_params = select_diverse_parameters(all_parameters, n_params)
                default_params = [p for p in all_parameters if p not in elicit_params]
            else:
                elicit_params = list(all_parameters)
                default_params = []

            # Fill defaults for non-elicited parameters
            default_priors = {p.name: get_default_prior(p) for p in default_params}

            # Compute data stats for feedback (if raw_data available)
            data_stats = (
                _compute_data_stats(raw_data)
                if raw_data is not None and not raw_data.is_empty()
                else {}
            )

            # Phase 1: Literature search (parallel, cached for retries)
            async def _search(param: ParameterSpec) -> tuple[str, list[dict]]:
                sources = await search_parameter_literature(param)
                return param.name, sources

            search_tasks = [_search(p) for p in elicit_params]
            search_results = await asyncio.gather(*search_tasks)
            literature_cache: dict[str, list[dict]] = dict(search_results)

            # Phase 2: Elicit priors (parallel)
            prior_results: dict[str, object] = {}

            async def _elicit(
                param: ParameterSpec,
                feedback: str | None = None,
            ) -> tuple[str, object]:
                sources = literature_cache[param.name]
                lit_context = format_literature_for_parameter(sources)
                result = await elicit_prior(
                    parameter=param,
                    question=question,
                    generate=gen_fn,
                    literature_context=lit_context,
                    literature_sources=sources,
                    feedback=feedback,
                )
                return param.name, result

            elicit_tasks = [_elicit(p) for p in elicit_params]
            elicit_results = await asyncio.gather(*elicit_tasks)
            for name, result in elicit_results:
                prior_results[name] = result

            # Build full priors dict
            def _build_priors() -> dict[str, object]:
                priors = {}
                for name, res in prior_results.items():
                    priors[name] = res.proposal
                for name, proposal in default_priors.items():
                    priors[name] = proposal
                return priors

            priors = _build_priors()

            # Phase 3: Validate prior predictive
            is_valid, validation_results, pp_samples = validate_prior_predictive(
                model_spec=model_spec_dict,
                priors=priors,
                raw_data=raw_data,
                causal_spec=causal_spec,
            )

            retries_used = 0

            # Phase 4: Retry loop (matches pipeline: max_prior_retries attempts)
            for attempt in range(MAX_PRIOR_RETRIES):
                if is_valid:
                    break

                retries_used = attempt + 1
                param_names = [p.name for p in elicit_params]
                failed_names = get_failed_parameters(
                    validation_results, param_names, causal_spec=causal_spec
                )

                # If no specific params identified, re-elicit all (matches pipeline)
                if not failed_names:
                    failed_names = list(prior_results.keys())

                # Re-elicit only failed params with feedback
                failed_params = [p for p in elicit_params if p.name in failed_names]
                re_elicit_tasks = []
                for param in failed_params:
                    prior_dict = prior_results[param.name].proposal.model_dump()
                    feedback = format_parameter_feedback(
                        param.name,
                        validation_results,
                        prior=prior_dict,
                        data_stats=data_stats if data_stats else None,
                    )
                    if feedback:
                        re_elicit_tasks.append(_elicit(param, feedback=feedback))

                if not re_elicit_tasks:
                    break

                re_results = await asyncio.gather(*re_elicit_tasks)
                for name, result in re_results:
                    prior_results[name] = result

                # Rebuild and re-validate
                priors = _build_priors()
                is_valid, validation_results, pp_samples = validate_prior_predictive(
                    model_spec=model_spec_dict,
                    priors=priors,
                    raw_data=raw_data,
                    causal_spec=causal_spec,
                )

            # Phase 5: Build SSM model (verify priors produce a working model)
            model_build_info = _try_build_model(model_spec_dict, priors, causal_spec)

            # Store results for scorer
            state.metadata["prior_results"] = {
                name: res.model_dump() for name, res in prior_results.items()
            }
            state.metadata["default_prior_names"] = list(default_priors.keys())
            state.metadata["validation_is_valid"] = is_valid
            state.metadata["validation_results"] = [r.model_dump() for r in validation_results]
            state.metadata["retries_used"] = retries_used
            state.metadata["model_build_info"] = model_build_info

            # Store pipeline-compatible priors dict for persistence
            state.metadata["priors"] = {
                name: (
                    res.proposal.model_dump()
                    if hasattr(res, "proposal")
                    else res.model_dump()
                )
                for name, res in prior_results.items()
            }
            for name, proposal in default_priors.items():
                state.metadata["priors"][name] = proposal.model_dump()

            # Build completion summary
            n_elicited = len(prior_results)
            n_defaults = len(default_priors)
            n_lit = sum(1 for r in prior_results.values() if r.literature_found)
            state.output.completion = (
                f"Elicited {n_elicited} priors ({n_lit} with literature), "
                f"{n_defaults} defaults. "
                f"Validation: {'PASS' if is_valid else 'FAIL'}"
                f"{' (after retry)' if retries_used > 0 and is_valid else ''}"
                f". Model build: {'OK' if model_build_info.get('model_built') else 'FAILED'}"
            )

            return state

        return solve

    return _solver()


def _try_build_model(
    model_spec: dict,
    priors: dict,
    causal_spec: dict | None,
) -> dict:
    """Try to build the SSM model from spec + priors (mirrors pipeline's build_model_task)."""
    try:
        import polars as pl

        from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
        from causal_ssm_agent.orchestrator.schemas_model import ModelSpec
        from causal_ssm_agent.utils.data import pivot_to_wide

        # Serialize priors to dicts
        priors_dict = {}
        for name, prior in priors.items():
            if hasattr(prior, "model_dump"):
                priors_dict[name] = prior.model_dump()
            else:
                priors_dict[name] = prior

        builder = SSMModelBuilder(
            model_spec=model_spec, priors=priors_dict, causal_spec=causal_spec
        )

        # Create minimal data for building
        spec_obj = ModelSpec.model_validate(model_spec)
        manifest_names = [lik.variable for lik in spec_obj.likelihoods]
        cols = {name: [0.0] * 10 for name in manifest_names}
        cols["time"] = list(range(10))
        X_wide = pl.DataFrame(cols).cast(dict.fromkeys(manifest_names, pl.Float64))

        builder.build_model(X_wide)

        return {"model_built": True}
    except Exception as e:
        logger.warning("SSM model build failed: %s", e)
        return {"model_built": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

# Expected distribution families per constraint
_EXPECTED_DISTRIBUTIONS: dict[str, set[str]] = {
    ParameterConstraint.POSITIVE: {"HalfNormal", "HalfCauchy", "Gamma", "Exponential", "LogNormal"},
    ParameterConstraint.UNIT_INTERVAL: {"Beta"},
    ParameterConstraint.CORRELATION: {"Uniform", "TruncatedNormal", "LKJCholesky"},
    ParameterConstraint.NONE: {"Normal", "Cauchy", "StudentT", "Laplace"},
}


def _score_single_parameter(prior_result: dict) -> tuple[int, list[str]]:
    """Score a single elicited parameter. Returns (points, detail_lines)."""
    points = 0
    details = []
    proposal = prior_result["proposal"]
    param_name = prior_result["parameter"]

    # 1. Distribution matches constraint (+2)
    # We need to infer constraint from the proposal context
    dist = proposal.get("distribution", "")
    # Check if distribution is a recognized valid name
    all_valid = set()
    for dists in _EXPECTED_DISTRIBUTIONS.values():
        all_valid |= dists
    if dist in all_valid:
        points += 2
        details.append(f"{param_name}: distribution '{dist}' is valid (+2)")
    else:
        details.append(f"{param_name}: distribution '{dist}' not in expected set (+0)")

    # 2. Params valid: finite, scale > 0 (+1)
    params = proposal.get("params", {})
    params_valid = True
    for k, v in params.items():
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            params_valid = False
            break
        if k in ("sigma", "scale", "beta", "alpha", "upper") and v <= 0:
            params_valid = False
            break
    if params_valid and params:
        points += 1
        details.append(f"{param_name}: params valid (+1)")
    else:
        details.append(f"{param_name}: params invalid (+0)")

    # 3. Literature found (+1)
    if prior_result.get("literature_found", False):
        points += 1
        details.append(f"{param_name}: literature found (+1)")
    else:
        details.append(f"{param_name}: no literature (+0)")

    # 4. Sources cited by LLM (+1)
    sources = proposal.get("sources", [])
    if sources:
        points += 1
        details.append(f"{param_name}: {len(sources)} source(s) cited (+1)")
    else:
        details.append(f"{param_name}: no sources cited (+0)")

    # 5. Reasoning present (+1)
    reasoning = proposal.get("reasoning", "")
    if len(reasoning) > 20:
        points += 1
        details.append(f"{param_name}: reasoning present (+1)")
    else:
        details.append(f"{param_name}: reasoning missing/short (+0)")

    return points, details


@scorer(metrics=[mean(), stderr()])
def prior_elicitation_scorer():
    """Score prior elicitation quality.

    Per-parameter scoring (max 6 pts each) + system-level validation bonus.
    Total score = sum of per-param points + validation bonus.
    """

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        prior_results = state.metadata.get("prior_results")
        if prior_results is None:
            return Score(
                value=0,
                answer="NO_RESULTS",
                explanation="Prior elicitation did not produce results",
            )

        # Per-parameter scoring
        total_points = 0
        max_per_param = 6
        all_details: list[str] = []
        n_elicited = len(prior_results)

        for _name, result in prior_results.items():
            pts, details = _score_single_parameter(result)
            total_points += pts
            all_details.extend(details)

        param_max = n_elicited * max_per_param

        # System-level validation bonus
        is_valid = state.metadata.get("validation_is_valid", False)
        retries_used = state.metadata.get("retries_used", 0)

        if is_valid and retries_used == 0:
            validation_bonus = 10
            all_details.append("Validation PASSED on first attempt (+10)")
        elif is_valid and retries_used > 0:
            validation_bonus = 5
            all_details.append("Validation PASSED after retry (+5)")
        else:
            validation_bonus = 0
            all_details.append("Validation FAILED (+0)")

        total_points += validation_bonus
        max_points = param_max + 10  # Max bonus is always 10

        explanation_lines = [
            f"Score: {total_points}/{max_points}",
            f"  Parameters elicited: {n_elicited}",
            f"  Defaults used: {len(state.metadata.get('default_prior_names', []))}",
            f"  Validation: {'PASS' if is_valid else 'FAIL'}"
            + (f" (retry {retries_used})" if retries_used > 0 else ""),
            "",
            *all_details,
        ]

        return Score(
            value=total_points,
            answer=f"{total_points}/{max_points}",
            explanation="\n".join(explanation_lines),
            metadata={
                "total_points": total_points,
                "max_points": max_points,
                "param_points": total_points - validation_bonus,
                "validation_bonus": validation_bonus,
                "n_elicited": n_elicited,
                "n_defaults": len(state.metadata.get("default_prior_names", [])),
                "validation_passed": is_valid,
                "retries_used": retries_used,
            },
        )

    return score


# ══════════════════════════════════════════════════════════════════════════════
# Task
# ══════════════════════════════════════════════════════════════════════════════


@task
def prior_elicitation_eval(
    questions: str | None = None,
    n_params: int | None = None,
):
    """Evaluate prior elicitation pipeline quality.

    Tests the full production code path: Exa literature search, LLM prior
    elicitation, and prior predictive validation with retry feedback loop.

    Args:
        questions: Optional comma-separated question selectors (e.g. "1")
        n_params: Max parameters to elicit (None = all). Limits Exa + LLM cost.
    """
    return Task(
        dataset=create_eval_dataset(questions=questions, n_params=n_params),
        solver=[prior_elicitation_solver()],
        scorer=prior_elicitation_scorer(),
    )
