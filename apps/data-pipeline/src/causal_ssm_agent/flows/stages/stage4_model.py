"""Stage 4: Model Specification & Prior Elicitation.

Orchestrator-Worker architecture with SSM grounding:
1. Orchestrator proposes ModelSpec
2. Exa literature search per parameter (run once, cached)
3. Workers elicit priors in parallel (one per parameter)
4. Prior predictive validation loop:
   - Validate priors
   - On failure, re-elicit only failed parameters with feedback
   - Max N retries, reusing cached Exa results
5. Build SSMModel (only if validation passes or retries exhausted)

See docs/modeling/functional_spec.md for design rationale.
"""

import polars as pl
from prefect import flow, task
from prefect.cache_policies import INPUTS

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
    cache_policy=INPUTS,
    persist_result=True,
    retries=2,
    retry_delay_seconds=5,
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


@task(retries=1, task_run_name="validate-priors")
def validate_priors_task(
    model_spec: dict,
    priors: dict[str, dict],
    raw_data: pl.DataFrame,
    causal_spec: dict | None = None,
) -> dict:
    """Validate priors via prior predictive sampling.

    Args:
        model_spec: Model specification dict
        priors: Prior proposals by parameter name
        raw_data: Raw timestamped data
        causal_spec: CausalSpec dict for DAG-constrained masks

    Returns:
        Validation result dict with is_valid and issues
    """
    try:
        from causal_ssm_agent.models.prior_predictive import validate_prior_predictive

        is_valid, results, raw_samples = validate_prior_predictive(
            model_spec, priors, raw_data, causal_spec=causal_spec
        )

        # Forward-simulate per-variable prior predictive observations
        pp_samples: dict[str, list[float]] = {}
        if is_valid and raw_samples:
            try:
                import jax.numpy as jnp
                import numpy as np

                from causal_ssm_agent.models.posterior_predictive import (
                    simulate_posterior_predictive,
                )
                from causal_ssm_agent.orchestrator.schemas_model import ModelSpec

                spec = (
                    ModelSpec.model_validate(model_spec)
                    if isinstance(model_spec, dict)
                    else model_spec
                )
                manifest_names = [lik.variable for lik in spec.likelihoods]
                manifest_dists = [lik.distribution.value for lik in spec.likelihoods]
                manifest_links = [lik.link.value for lik in spec.likelihoods]

                y_sim = simulate_posterior_predictive(
                    raw_samples,
                    times=jnp.arange(30, dtype=jnp.float32),
                    manifest_dists=manifest_dists,
                    manifest_links=manifest_links,
                    n_subsample=100,
                )
                # y_sim: (n_subsample, T, n_manifest) → flatten to per-variable lists
                y_np = np.asarray(y_sim)
                for j, name in enumerate(manifest_names):
                    col = y_np[:, :, j].flatten()
                    # Filter out NaN/Inf from unstable draws
                    col = col[np.isfinite(col)]
                    pp_samples[name] = col.tolist()
            except Exception as e:
                logger.warning("Prior predictive simulation failed: %s", e)

        return {
            "is_valid": is_valid,
            "results": [r.model_dump() for r in results],
            "issues": [r.issue for r in results if not r.is_valid and r.issue],
            "prior_predictive_samples": pp_samples,
        }
    except Exception as e:
        return {
            "is_valid": False,
            "results": [],
            "issues": [f"Prior validation error: {e}"],
            "prior_predictive_samples": {},
        }


@task(task_run_name="compile-ssm-model")
def compile_model_task(
    model_spec: dict,
    priors: dict[str, dict],
    raw_data: pl.DataFrame,
    causal_spec: dict | None = None,
) -> dict:
    """Compile and verify an executable SSM artifact.

    Args:
        model_spec: Model specification
        priors: Prior proposals
        raw_data: Raw timestamped data (indicator, value, timestamp)
        causal_spec: CausalSpec dict for DAG-constrained masks

    Returns:
        Dict with compile/build status and serialized artifact
    """
    from causal_ssm_agent.models.ssm_builder import build_ssm_builder
    from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact

    try:
        compiled_ssm = compile_ssm_artifact(model_spec, priors, causal_spec=causal_spec)
        builder = build_ssm_builder(
            raw_data=raw_data,
            compiled_ssm=compiled_ssm,
        )

        return {
            "model_built": True,
            "model_type": builder._model_type,
            "version": builder.version,
            "compiled_ssm": compiled_ssm,
        }

    except NotImplementedError:
        return {
            "model_built": False,
            "error": "SSM implementation not available",
        }
    except Exception as e:
        return {
            "model_built": False,
            "error": str(e),
        }


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
    5. Compile the executable SSM artifact (only when validation passes or retries exhausted)

    Args:
        causal_spec: Full CausalSpec dict
        question: Research question
        raw_data: Raw timestamped data (indicator, value, timestamp)
        enable_literature: Whether to search Exa for literature
        max_prior_retries: Maximum validation retry attempts

    Returns:
        Stage 4 result dict with model_spec, priors, validation
    """
    from prefect.utilities.annotations import unmapped

    from causal_ssm_agent.models.prior_predictive import (
        _compute_data_stats,
        format_parameter_feedback,
        get_failed_parameters,
    )
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.workers.schemas_prior import PriorValidationResult

    config = get_config()
    if max_prior_retries is None:
        max_prior_retries = config.pipeline.max_prior_retries
    paraphrasing = config.stage4_prior_elicitation.paraphrasing
    n_paraphrases = paraphrasing.n_paraphrases if paraphrasing.enabled else 1

    # 1. Orchestrator proposes model specification. Stage 1b owns structural
    # validation, so stage 4 only performs a single compile-time assertion.
    # Retry up to 2 times if the LLM proposes unsupported distributions/structures.
    from causal_ssm_agent.models.ssm_compiler import trial_compile_model_spec
    from causal_ssm_agent.utils.identifiability import inject_marginalized_correlations

    max_spec_attempts = 3
    compile_error = None
    llm_trace = None
    for spec_attempt in range(max_spec_attempts):
        model_spec = await propose_model_task(causal_spec, question, raw_data)
        llm_trace = model_spec.pop("llm_trace", None)

        inject_marginalized_correlations(model_spec, causal_spec)

        compile_error = trial_compile_model_spec(model_spec, causal_spec)
        if compile_error is None:
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

    parameter_specs = model_spec.get("parameters", [])

    logger.info("Stage 4: %d parameters", len(parameter_specs))

    # Build a lookup from parameter name -> spec dict
    param_spec_by_name = {ps.get("name", f"param_{i}"): ps for i, ps in enumerate(parameter_specs)}

    # 2. Exa literature search per parameter (run once, cached for retries)
    #    task.map() fans out all parameters; concurrency limit inside
    #    the task caps how many actually hit the API simultaneously.
    if enable_literature:
        literature_futures = search_literature_task.map(parameter_specs)
        literature_by_name = {}
        for i, (ps, future) in enumerate(zip(parameter_specs, literature_futures)):
            name = ps.get("name", f"param_{i}")
            try:
                literature_by_name[name] = future.result()
            except Exception as e:
                logger.warning("Literature search failed for %s: %s", name, e)
                literature_by_name[name] = {"sources": [], "formatted": ""}
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

    # Smoke-test: compile with actual (not default) priors before the expensive
    # prior predictive sampling.  Catches structural issues early (unrecognized
    # distributions, param mismatches, constraint violations).
    from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact as _compile

    try:
        _compile(model_spec, priors, causal_spec=causal_spec)
    except Exception as e:
        logger.warning("Post-elicitation compile check failed: %s", e)

    # Compute data stats once for feedback messages
    data_stats = (
        _compute_data_stats(raw_data) if raw_data is not None and not raw_data.is_empty() else {}
    )

    # 4. Validation loop
    validation_result = None
    for attempt in range(max_prior_retries + 1):
        validation_result = validate_priors_task(
            model_spec, priors, raw_data, causal_spec=causal_spec
        )

        if validation_result.get("is_valid", False):
            logger.info("Prior validation passed on attempt %d", attempt + 1)
            break

        if attempt >= max_prior_retries:
            logger.warning(
                "Prior validation failed after %d attempts. Proceeding with best priors.",
                max_prior_retries + 1,
            )
            break

        # Identify which parameters need re-elicitation
        vr_objects = [
            PriorValidationResult.model_validate(r) for r in validation_result.get("results", [])
        ]
        failed_param_names = get_failed_parameters(
            vr_objects, list(priors.keys()), causal_spec=causal_spec
        )

        # If validation failed but no specific parameters identified (e.g., validator
        # exception returned empty results), treat as global failure: re-elicit all.
        if not failed_param_names:
            if not validation_result.get("is_valid", False):
                logger.warning(
                    "Validation failed with no per-parameter results; re-eliciting all parameters"
                )
                failed_param_names = list(priors.keys())
            else:
                break

        logger.info(
            "Attempt %d: re-eliciting %d failed parameters: %s",
            attempt + 1,
            len(failed_param_names),
            failed_param_names,
        )

        # Build per-parameter feedback
        feedbacks = {}
        for param_name in failed_param_names:
            feedbacks[param_name] = format_parameter_feedback(
                parameter_name=param_name,
                results=vr_objects,
                prior=priors.get(param_name),
                data_stats=data_stats,
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

    # 5. Compile the executable artifact (only after validation loop)
    compile_task = compile_model_task(model_spec, priors, raw_data, causal_spec=causal_spec)
    model_result = compile_task.result() if hasattr(compile_task, "result") else compile_task
    compiled_ssm = model_result.pop("compiled_ssm", None)

    result = {
        "model_spec": model_spec,
        "priors": priors,
        "validation": validation_result,
        "model_info": model_result,
        "is_valid": validation_result.get("is_valid", False) if validation_result else False,
        "causal_spec": causal_spec,
        "prior_predictive_samples": (validation_result or {}).get("prior_predictive_samples", {}),
    }
    if compiled_ssm is not None:
        result["_compiled_ssm"] = compiled_ssm
    if llm_trace is not None:
        result["llm_trace"] = llm_trace
    return result
