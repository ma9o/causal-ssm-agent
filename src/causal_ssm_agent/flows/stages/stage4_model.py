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

import logging

import polars as pl
from prefect import flow, task

logger = logging.getLogger(__name__)


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


@task(retries=2, retry_delay_seconds=10, task_run_name="propose-model-spec")
def propose_model_task(
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
    import asyncio

    from inspect_ai.model import get_model

    from causal_ssm_agent.orchestrator.stage4_orchestrator import (
        propose_model_spec,
    )
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.llm import make_orchestrator_generate_fn

    async def run():
        config = get_config()
        model = get_model(config.stage4_prior_elicitation.model)
        generate = make_orchestrator_generate_fn(model)

        data_summary = build_raw_data_summary(raw_data)

        result = await propose_model_spec(
            causal_spec=causal_spec,
            data_summary=data_summary,
            question=question,
            generate=generate,
        )

        return result.model_spec.model_dump()

    return asyncio.run(run())


@task(retries=2, retry_delay_seconds=5, task_run_name="search-literature-{parameter_spec[name]}")
def search_literature_task(
    parameter_spec: dict,
) -> dict:
    """Search Exa for literature relevant to a parameter.

    Run once per parameter; results are cached for reuse across retry loops.

    Args:
        parameter_spec: ParameterSpec as dict

    Returns:
        Dict with 'sources' (raw Exa results) and 'formatted' (prompt string)
    """
    import asyncio

    from causal_ssm_agent.orchestrator.schemas_model import ParameterSpec
    from causal_ssm_agent.workers.prior_research import search_parameter_literature
    from causal_ssm_agent.workers.prompts.prior_research import (
        format_literature_for_parameter,
    )

    async def run():
        param = ParameterSpec.model_validate(parameter_spec)
        sources = await search_parameter_literature(param)
        formatted = format_literature_for_parameter(sources)
        return {"sources": sources, "formatted": formatted}

    return asyncio.run(run())


@task(retries=2, retry_delay_seconds=5, task_run_name="elicit-prior-{parameter_spec[name]}")
def elicit_prior_task(
    parameter_spec: dict,
    question: str,
    literature: dict,
    n_paraphrases: int = 1,
    feedback: str | None = None,
) -> dict:
    """Elicit a prior for a single parameter using LLM.

    Uses pre-fetched literature context (no Exa call). Accepts optional
    feedback from a previous failed validation for re-elicitation.

    Args:
        parameter_spec: ParameterSpec as dict
        question: Research question
        literature: Cached literature dict from search_literature_task
        n_paraphrases: Number of paraphrased prompts
        feedback: Validation feedback from previous attempt

    Returns:
        PriorProposal as dict
    """
    import asyncio

    from inspect_ai.model import get_model

    from causal_ssm_agent.orchestrator.schemas_model import ParameterSpec
    from causal_ssm_agent.utils.config import get_config
    from causal_ssm_agent.utils.llm import make_worker_generate_fn
    from causal_ssm_agent.workers.prior_research import (
        elicit_prior,
        get_default_prior,
    )

    async def run():
        config = get_config()
        worker_model = config.stage4_prior_elicitation.worker_model or config.stage2_workers.model
        model = get_model(worker_model)
        generate = make_worker_generate_fn(model)

        param = ParameterSpec.model_validate(parameter_spec)

        try:
            result = await elicit_prior(
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

    return asyncio.run(run())


@task(retries=1, task_run_name="validate-priors")
def validate_priors_task(
    model_spec: dict,
    priors: dict[str, dict],
    raw_data: pl.DataFrame,
) -> dict:
    """Validate priors via prior predictive sampling.

    Args:
        model_spec: Model specification dict
        priors: Prior proposals by parameter name
        raw_data: Raw timestamped data

    Returns:
        Validation result dict with is_valid and issues
    """
    try:
        from causal_ssm_agent.models.prior_predictive import validate_prior_predictive

        is_valid, results = validate_prior_predictive(model_spec, priors, raw_data)
        return {
            "is_valid": is_valid,
            "results": [r.model_dump() for r in results],
            "issues": [r.issue for r in results if not r.is_valid and r.issue],
        }
    except Exception as e:
        return {
            "is_valid": False,
            "results": [],
            "issues": [f"Prior validation error: {e}"],
        }


@task(task_run_name="build-ssm-model")
def build_model_task(
    model_spec: dict,
    priors: dict[str, dict],
    raw_data: pl.DataFrame,
) -> dict:
    """Build SSMModelBuilder from spec and priors.

    Args:
        model_spec: Model specification
        priors: Prior proposals
        raw_data: Raw timestamped data (indicator, value, timestamp)

    Returns:
        Dict with model_built status and builder info
    """
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

    try:
        builder = SSMModelBuilder(model_spec=model_spec, priors=priors)

        # Convert raw data to wide format for model building
        # SSM expects: time column + indicator columns
        if raw_data.is_empty():
            return {
                "model_built": False,
                "error": "No data available",
            }

        # Pivot data: rows=time points, columns=indicators
        time_col = "time_bucket" if "time_bucket" in raw_data.columns else "timestamp"
        wide_data = (
            raw_data.with_columns(pl.col("value").cast(pl.Float64, strict=False))
            .pivot(on="indicator", index=time_col, values="value")
            .sort(time_col)
        )

        X = wide_data.to_pandas()

        # Rename time column to "time" for SSM
        if time_col in X.columns:
            X = X.rename(columns={time_col: "time"})

        builder.build_model(X)

        return {
            "model_built": True,
            "model_type": builder._model_type,
            "version": builder.version,
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
def stage4_orchestrated_flow(
    causal_spec: dict,
    question: str,
    raw_data: pl.DataFrame,
    enable_literature: bool = True,
    max_prior_retries: int | None = None,
) -> dict:
    """Stage 4 orchestrated flow with validation-driven prior elicitation.

    1. Orchestrator proposes model specification (with syntax validation loop)
    2. Exa literature search per parameter (run once, cached)
    3. LLM elicits priors in parallel
    4. Prior predictive validation loop:
       - Validate all priors
       - On failure, re-elicit only failed parameters in parallel
       - Feed validation issues + data scale back to LLM
       - Max N retries, reusing cached Exa results
    5. Build SSMModel (only when validation passes or retries exhausted)

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

    # 1. Orchestrator proposes model specification
    model_spec = propose_model_task(causal_spec, question, raw_data)
    parameter_specs = model_spec.get("parameters", [])

    # Build a lookup from parameter name -> spec dict
    param_spec_by_name = {ps.get("name", f"param_{i}"): ps for i, ps in enumerate(parameter_specs)}

    # 2. Exa literature search per parameter (run once, cached for retries)
    if enable_literature:
        literature_results = search_literature_task.map(parameter_specs)
        literature_by_name = {}
        for ps, lit in zip(parameter_specs, literature_results):
            name = ps.get("name", "unknown")
            literature_by_name[name] = lit.result() if hasattr(lit, "result") else lit
    else:
        literature_by_name = {
            ps.get("name", "unknown"): {"sources": [], "formatted": ""} for ps in parameter_specs
        }

    # 3. Initial LLM elicitation (all parameters in parallel)
    initial_results = elicit_prior_task.map(
        parameter_specs,
        question=unmapped(question),
        literature=[literature_by_name[ps.get("name", "unknown")] for ps in parameter_specs],
        n_paraphrases=unmapped(n_paraphrases),
    )

    priors = {}
    for ps, result in zip(parameter_specs, initial_results):
        name = ps.get("name", "unknown")
        priors[name] = result.result() if hasattr(result, "result") else result

    # Compute data stats once for feedback messages
    data_stats = (
        _compute_data_stats(raw_data) if raw_data is not None and not raw_data.is_empty() else {}
    )

    # 4. Validation loop
    validation_result = None
    for attempt in range(max_prior_retries + 1):
        validation = validate_priors_task(model_spec, priors, raw_data)
        validation_result = validation.result() if hasattr(validation, "result") else validation

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
        failed_param_names = get_failed_parameters(vr_objects, list(priors.keys()))

        if not failed_param_names:
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

        # Re-elicit only failed parameters in parallel
        failed_specs = [param_spec_by_name[n] for n in failed_param_names]
        failed_literature = [
            literature_by_name.get(n, {"sources": [], "formatted": ""}) for n in failed_param_names
        ]
        failed_feedbacks = [feedbacks[n] for n in failed_param_names]

        re_results = elicit_prior_task.map(
            failed_specs,
            question=unmapped(question),
            literature=failed_literature,
            n_paraphrases=unmapped(n_paraphrases),
            feedback=failed_feedbacks,
        )

        # Merge re-elicited priors back
        for name, result in zip(failed_param_names, re_results):
            priors[name] = result.result() if hasattr(result, "result") else result

    # 5. Build SSMModel (only after validation loop)
    model_info = build_model_task(model_spec, priors, raw_data)
    model_result = model_info.result() if hasattr(model_info, "result") else model_info

    return {
        "model_spec": model_spec,
        "priors": priors,
        "validation": validation_result,
        "model_info": model_result,
        "is_valid": validation_result.get("is_valid", False) if validation_result else False,
        "raw_data": raw_data,  # Pass through for Stage 5
    }
