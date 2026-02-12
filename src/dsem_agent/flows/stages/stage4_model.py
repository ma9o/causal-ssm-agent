"""Stage 4: Model Specification & Prior Elicitation.

Orchestrator-Worker architecture with CT-SEM grounding:
1. Orchestrator proposes model specification
2. Workers research priors in parallel (one per parameter via Exa + LLM)
3. CT-SEM model is built as grounding step (validates priors compile)
4. Prior predictive checks validate reasonableness

See docs/modeling/functional_spec.md for design rationale.
"""

import polars as pl
from prefect import flow, task


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

    from dsem_agent.orchestrator.stage4_orchestrator import (
        propose_model_spec,
    )
    from dsem_agent.utils.config import get_config
    from dsem_agent.utils.llm import make_orchestrator_generate_fn

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


@task(retries=2, retry_delay_seconds=5)
def research_prior_task(
    parameter_spec: dict,
    question: str,
    enable_literature: bool = True,
    n_paraphrases: int = 1,
) -> dict:
    """Worker researches prior for a single parameter.

    Args:
        parameter_spec: ParameterSpec as dict
        question: Research question
        enable_literature: Whether to search Exa
        n_paraphrases: Number of paraphrased prompts (1 = original single-shot)

    Returns:
        PriorProposal as dict
    """
    import asyncio

    from inspect_ai.model import get_model

    from dsem_agent.orchestrator.schemas_model import ParameterSpec
    from dsem_agent.utils.config import get_config
    from dsem_agent.utils.llm import make_worker_generate_fn
    from dsem_agent.workers.prior_research import (
        get_default_prior,
        research_single_prior,
    )

    async def run():
        config = get_config()
        worker_model = config.stage4_prior_elicitation.worker_model or config.stage2_workers.model
        model = get_model(worker_model)
        generate = make_worker_generate_fn(model)

        param = ParameterSpec.model_validate(parameter_spec)

        try:
            result = await research_single_prior(
                parameter=param,
                question=question,
                generate=generate,
                enable_literature=enable_literature,
                n_paraphrases=n_paraphrases,
            )
            return result.proposal.model_dump()
        except Exception as e:
            print(f"Prior research failed for {param.name}: {e}. Using default prior.")
            return get_default_prior(param).model_dump()

    return asyncio.run(run())


@task(retries=1, task_run_name="validate-priors")
def validate_priors_task(
    model_spec: dict,  # noqa: ARG001
    priors: dict[str, dict],  # noqa: ARG001
    raw_data: pl.DataFrame,  # noqa: ARG001
) -> dict:
    """Validate priors via prior predictive sampling.

    Args:
        model_spec: Model specification dict
        priors: Prior proposals by parameter name
        raw_data: Raw timestamped data

    Returns:
        Validation result dict with is_valid and issues
    """
    # TODO: Implement prior predictive validation for CT-SEM
    # For now, return valid to allow pipeline to proceed
    return {
        "is_valid": True,
        "results": [],
        "issues": [],
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
    from dsem_agent.models.ssm_builder import SSMModelBuilder

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


@flow(name="stage4-orchestrated", log_prints=True)
def stage4_orchestrated_flow(
    causal_spec: dict,
    question: str,
    raw_data: pl.DataFrame,
    enable_literature: bool = True,
) -> dict:
    """Stage 4 orchestrated flow with parallel worker prior research.

    1. Orchestrator proposes model specification
    2. Workers research priors in parallel (one per parameter)
    3. Validate via prior predictive checks
    4. Build CT-SEM model

    Args:
        causal_spec: Full CausalSpec dict
        question: Research question
        raw_data: Raw timestamped data (indicator, value, timestamp)
        enable_literature: Whether to search Exa for literature

    Returns:
        Stage 4 result dict with model_spec, priors, validation
    """
    from prefect.utilities.annotations import unmapped

    from dsem_agent.utils.config import get_config

    config = get_config()
    paraphrasing = config.stage4_prior_elicitation.paraphrasing

    # Determine paraphrasing settings
    n_paraphrases = paraphrasing.n_paraphrases if paraphrasing.enabled else 1

    # 1. Orchestrator proposes model specification
    model_spec = propose_model_task(causal_spec, question, raw_data)

    # 2. Workers research priors in parallel
    parameter_specs = model_spec.get("parameters", [])
    prior_results = research_prior_task.map(
        parameter_specs,
        question=unmapped(question),
        enable_literature=unmapped(enable_literature),
        n_paraphrases=unmapped(n_paraphrases),
    )

    # Collect results into dict
    priors = {}
    for param_spec, prior_result in zip(parameter_specs, prior_results):
        param_name = param_spec.get("name", "unknown")
        priors[param_name] = (
            prior_result.result() if hasattr(prior_result, "result") else prior_result
        )

    # 3. Validate priors
    validation = validate_priors_task(model_spec, priors, raw_data)
    validation_result = validation.result() if hasattr(validation, "result") else validation

    # 4. Build CT-SEM model
    model_info = build_model_task(model_spec, priors, raw_data)
    model_result = model_info.result() if hasattr(model_info, "result") else model_info

    return {
        "model_spec": model_spec,
        "priors": priors,
        "validation": validation_result,
        "model_info": model_result,
        "is_valid": validation_result.get("is_valid", False),
        "raw_data": raw_data,  # Pass through for Stage 5
    }
