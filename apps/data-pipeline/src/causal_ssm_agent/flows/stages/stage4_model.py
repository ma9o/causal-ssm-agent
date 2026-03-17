"""Stage 4: Model Specification & Prior Elicitation.

Orchestrator-Worker architecture with SSM grounding:
1. Orchestrator proposes ModelSpec
2. Exa literature search per parameter (run once, cached)
3. Workers elicit priors in parallel (one per parameter)
4. Prior predictive validation loop:
   - Validate priors
   - On failure, re-elicit only failed parameters with feedback
   - Max N retries, reusing cached Exa results
5. Return authored stage-4 state; the pipeline materializer derives validation
   and compiled artifacts once, in one place.

See docs/modeling/functional_spec.md for design rationale.
"""

import polars as pl
from prefect import flow, task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.utils.litellm_client import RpmLimiter, set_limiter

from .. import get_prefect_logger

logger = get_prefect_logger(__name__)


def build_raw_data_summary(raw_data: pl.DataFrame) -> str:
    """Build a summary of data for the orchestrator.

    Args:
        raw_data: DataFrame with columns: indicator, value, and either
            timestamp (raw) or time_bucket (aggregated).

    Returns:
        Text summary of the data
    """
    if raw_data.is_empty():
        return "No data available."

    time_col = "time_bucket" if "time_bucket" in raw_data.columns else "timestamp"
    lines = [f"Data Summary (observations, time column: {time_col}):"]

    # Overall stats
    n_obs = len(raw_data)
    lines.append(f"  Total observations: {n_obs}")

    # Per-indicator stats
    indicator_stats = (
        raw_data.group_by("indicator")
        .agg(
            [
                pl.col("value").cast(pl.Float64, strict=False).count().alias("n_obs"),
                pl.col("value").cast(pl.Float64, strict=False).mean().alias("mean"),
                pl.col("value").cast(pl.Float64, strict=False).std().alias("std"),
            ]
        )
        .sort("indicator")
    )

    lines.append("  Per indicator:")
    for row in indicator_stats.iter_rows(named=True):
        mean_str = f"{row['mean']:.2f}" if row["mean"] is not None else "N/A"
        std_str = f"{row['std']:.2f}" if row["std"] is not None else "N/A"
        lines.append(f"    {row['indicator']}: n={row['n_obs']}, mean={mean_str}, std={std_str}")

    return "\n".join(lines)


@task(
    cache_policy=INPUTS,
    persist_result=True,
    retries=2,
    retry_delay_seconds=10,
    task_run_name="propose-model-spec",
)
async def propose_model_task(
    causal_spec: dict,
    question: str,
    raw_data: pl.DataFrame,
) -> dict:
    """Orchestrator proposes model specification.

    Args:
        causal_spec: Full CausalSpec dict
        question: Research question
        raw_data: Raw timestamped data (indicator, value, timestamp)

    Returns:
        ModelSpec as dict
    """
    from causal_ssm_agent.orchestrator.stage4_orchestrator import (
        propose_model_spec,
    )
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.llm import LLMStageContext

    config = get_config()
    async with LLMStageContext("stage-4") as ctx:
        generate = ctx.make_generate(config.stage4_prior_elicitation.model)

        data_summary = build_raw_data_summary(raw_data)

        result = await propose_model_spec(
            causal_spec=causal_spec,
            data_summary=data_summary,
            question=question,
            generate=generate,
        )

        return ctx.finalize(result.model_spec.model_dump())


@task(
    retries=3,
    retry_delay_seconds=5,
    retry_jitter_factor=1.0,
    task_run_name="search-literature-{parameter_spec[name]}",
)
async def search_literature_task(
    parameter_spec: dict,
) -> dict:
    """Search Exa for literature relevant to a parameter.

    Run once per parameter; results are cached for reuse across retry loops.

    Args:
        parameter_spec: ParameterSpec as dict

    Returns:
        Dict with 'sources' (raw Exa results) and 'formatted' (prompt string)
    """
    from causal_ssm_agent.orchestrator.schemas_model import ParameterSpec
    from causal_ssm_agent.workers.prior_research import search_parameter_literature
    from causal_ssm_agent.workers.prompts.prior_research import (
        format_literature_for_parameter,
    )

    param = ParameterSpec.model_validate(parameter_spec)
    sources = await search_parameter_literature(param)
    formatted = format_literature_for_parameter(sources)
    return {"sources": sources, "formatted": formatted}


@task(
    cache_policy=INPUTS,
    persist_result=True,
    retries=2,
    retry_delay_seconds=5,
    task_run_name="elicit-prior-{parameter_spec[name]}",
)
async def elicit_prior_task(
    parameter_spec: dict,
    question: str,
    literature: dict,
    n_paraphrases: int = 1,
    feedback: str | None = None,
    model_spec: dict | None = None,
    causal_spec: dict | None = None,
    raw_data: pl.DataFrame | None = None,
    current_priors: dict | None = None,
) -> dict:
    """Elicit a prior for a single parameter using LLM.

    Dispatches to:
    - ``run_prior_elicitation`` (fat tool, self-correcting) for n_paraphrases=1
    - ``elicit_prior`` (paraphrased, GMM aggregation) for n_paraphrases>1

    Args:
        parameter_spec: ParameterSpec as dict
        question: Research question
        literature: Cached literature dict from search_literature_task
        n_paraphrases: Number of paraphrased prompts
        feedback: Validation feedback from previous attempt
        model_spec: Current model spec (enables compile + PP in fat tool)
        causal_spec: CausalSpec dict
        raw_data: Polars DataFrame for prior predictive checks
        current_priors: Existing priors for other parameters

    Returns:
        PriorProposal as dict
    """
    from causal_ssm_agent.orchestrator.schemas_model import ParameterSpec
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.llm import make_generate_fn
    from causal_ssm_agent.workers.prior_research import (
        elicit_prior_paraphrased,
        get_default_prior,
        run_prior_elicitation,
    )

    config = get_config()
    worker_model = (
        config.stage4_prior_elicitation.worker_model or config.stage4_prior_elicitation.model
    )
    generate = make_generate_fn(worker_model)

    param = ParameterSpec.model_validate(parameter_spec)

    try:
        if n_paraphrases <= 1:
            # Fat tool path: self-correcting loop with search + validate
            return await run_prior_elicitation(
                parameter=param,
                question=question,
                generate=generate,
                literature_context=literature.get("formatted", ""),
                feedback=feedback,
                model_spec=model_spec,
                current_priors=current_priors,
                raw_data=raw_data,
                causal_spec=causal_spec,
            )
        else:
            # Paraphrased path: N independent prompts, GMM aggregation
            result = await elicit_prior_paraphrased(
                parameter=param,
                question=question,
                generate=generate,
                literature_context=literature.get("formatted", ""),
                literature_sources=literature.get("sources", []),
                feedback=feedback,
                n_paraphrases=n_paraphrases,
            )
            return result.proposal.model_dump()
    except Exception as e:
        logger.warning("Prior elicitation failed for %s: %s. Using default.", param.name, e)
        return get_default_prior(param).model_dump()


@flow(name="stage4-orchestrated", log_prints=True, persist_result=True, result_serializer="json")
async def stage4_orchestrated_flow(
    causal_spec: dict,
    question: str,
    raw_data: pl.DataFrame,
    enable_literature: bool = True,
    max_prior_retries: int | None = None,
) -> dict:
    """Stage 4 orchestrated flow with validation-driven prior elicitation.

    1. Orchestrator proposes model specification (with syntax validation)
    2. Exa literature search per parameter (run once, cached)
    3. LLM elicits priors in parallel
    4. Prior predictive validation loop:
       - Validate all priors
       - On failure, re-elicit only failed parameters in parallel
       - Feed validation issues + data scale back to LLM
       - Max N retries, reusing cached Exa results
    5. Return the authored stage-4 state; the registry materializer derives the
       executable/runtime payload once after orchestration completes

    Args:
        causal_spec: Full CausalSpec dict
        question: Research question
        raw_data: Raw timestamped data (indicator, value, timestamp)
        enable_literature: Whether to search Exa for literature
        max_prior_retries: Maximum validation retry attempts

    Returns:
        Authored Stage 4 state (model_spec + priors + provenance). The pipeline
        materializer derives validation/model artifacts from this state.
    """
    from prefect.utilities.annotations import unmapped

    from causal_ssm_agent.models.prior_predictive import compute_data_stats
    from causal_ssm_agent.utils.config import get_config

    from .stage4_assembly import (
        build_retry_feedback,
        build_stage4_authored_state,
        validate_assembly,
    )

    config = get_config()
    if max_prior_retries is None:
        max_prior_retries = config.pipeline.max_prior_retries
    paraphrasing = config.stage4_prior_elicitation.paraphrasing
    n_paraphrases = paraphrasing.n_paraphrases if paraphrasing.enabled else 1

    # 1. Orchestrator proposes model specification. Stage 1b owns structural
    # validation, so stage 4 only performs a single compile-time assertion.
    # Retry up to 2 times if the LLM proposes unsupported distributions/structures.
    max_spec_attempts = 3
    compile_error = None
    llm_trace = None
    model_spec = None
    for spec_attempt in range(max_spec_attempts):
        proposed_spec = await propose_model_task(causal_spec, question, raw_data)
        llm_trace = proposed_spec.pop("llm_trace", None)

        validation = validate_assembly(proposed_spec, None, None, causal_spec)
        compile_error = validation.compile_error
        if validation.compile_ok and validation.normalized_model_spec is not None:
            model_spec = validation.normalized_model_spec
            break
        logger.warning(
            "Stage 4: model spec attempt %d/%d failed compilation: %s",
            spec_attempt + 1,
            max_spec_attempts,
            compile_error,
        )
    if compile_error is not None:
        raise ValueError(
            f"Stage 4 model spec failed compilation after {max_spec_attempts} attempts: {compile_error}"
        )
    assert model_spec is not None

    parameter_specs = model_spec.get("parameters", [])

    logger.info("Stage 4: %d parameters", len(parameter_specs))

    # Build a lookup from parameter name -> spec dict
    param_spec_by_name = {ps.get("name", f"param_{i}"): ps for i, ps in enumerate(parameter_specs)}

    # 2. Exa literature search per parameter (rate-limited to 8 req/s)
    if enable_literature:
        set_limiter("exa", RpmLimiter(max_requests=8, window_seconds=1.0))
        try:
            literature_futures = search_literature_task.map(parameter_specs)
            literature_by_name = {}
            for i, (ps, future) in enumerate(zip(parameter_specs, literature_futures)):
                name = ps.get("name", f"param_{i}")
                try:
                    literature_by_name[name] = future.result()
                except Exception as e:
                    logger.warning("Literature search failed for %s: %s", name, e)
                    literature_by_name[name] = {"sources": [], "formatted": ""}
        finally:
            set_limiter("exa", None)
    else:
        literature_by_name = {
            ps.get("name", f"param_{i}"): {"sources": [], "formatted": ""}
            for i, ps in enumerate(parameter_specs)
        }

    # 3. Initial LLM elicitation (all parameters via task.map, concurrency-limited)
    initial_futures = elicit_prior_task.map(
        parameter_specs,
        question=unmapped(question),
        literature=[
            literature_by_name[ps.get("name", f"param_{i}")] for i, ps in enumerate(parameter_specs)
        ],
        n_paraphrases=unmapped(n_paraphrases),
        model_spec=unmapped(model_spec),
        causal_spec=unmapped(causal_spec),
        raw_data=unmapped(raw_data),
    )

    initial_results = initial_futures.result()
    priors = {}
    for i, (ps, result) in enumerate(zip(parameter_specs, initial_results)):
        name = ps.get("name", f"param_{i}")
        priors[name] = result

    # Compute data stats once for feedback messages
    data_stats = (
        compute_data_stats(raw_data) if raw_data is not None and not raw_data.is_empty() else {}
    )

    # 4. Validation loop
    validation = None
    validation_retries: list[dict[str, object]] = []
    for attempt in range(max_prior_retries + 1):
        validation = validate_assembly(
            model_spec,
            priors,
            raw_data,
            causal_spec,
        )
        if validation.is_valid:
            logger.info("Prior validation passed on attempt %d", attempt + 1)
            break

        if attempt >= max_prior_retries:
            logger.warning(
                "Prior validation failed after %d attempts. Proceeding with best priors.",
                max_prior_retries + 1,
            )
            break

        failed_param_names, feedbacks, global_summary = build_retry_feedback(
            validation,
            priors,
            causal_spec=causal_spec,
            data_stats=data_stats,
        )

        # If validation failed but no specific parameters identified (e.g., validator
        # exception returned empty results), treat as global failure: re-elicit all.
        if not failed_param_names:
            if not validation.is_valid:
                logger.warning(
                    "Validation failed with no per-parameter results; re-eliciting all parameters"
                )
                failed_param_names = list(priors.keys())
            else:
                break
            feedbacks = dict.fromkeys(failed_param_names, "")

        if global_summary:
            logger.warning(
                "Attempt %d: global validation failure; skipping prior re-elicitation.\n%s",
                attempt + 1,
                global_summary,
            )
            break

        logger.info(
            "Attempt %d: re-eliciting %d failed parameters: %s",
            attempt + 1,
            len(failed_param_names),
            failed_param_names,
        )
        validation_retries.append(
            {
                "attempt": attempt + 1,
                "failed_params": failed_param_names,
                "feedback": "\n\n".join(
                    feedbacks[name] for name in failed_param_names if feedbacks.get(name)
                ),
            }
        )

        # Re-elicit only failed parameters (concurrency-limited via task.map)
        failed_specs = [param_spec_by_name[n] for n in failed_param_names]
        failed_literature = [
            literature_by_name.get(n, {"sources": [], "formatted": ""}) for n in failed_param_names
        ]
        failed_feedbacks = [feedbacks[n] for n in failed_param_names]

        re_futures = elicit_prior_task.map(
            failed_specs,
            question=unmapped(question),
            literature=failed_literature,
            n_paraphrases=unmapped(n_paraphrases),
            feedback=failed_feedbacks,
            model_spec=unmapped(model_spec),
            causal_spec=unmapped(causal_spec),
            raw_data=unmapped(raw_data),
            current_priors=unmapped(priors),
        )

        # Merge re-elicited priors back
        re_results = re_futures.result()
        for name, result in zip(failed_param_names, re_results):
            priors[name] = result

    assert validation is not None
    return build_stage4_authored_state(
        model_spec=model_spec,
        priors=priors,
        validation_retries=validation_retries or None,
        llm_trace=llm_trace,
        assembly_validation=validation,
    )
