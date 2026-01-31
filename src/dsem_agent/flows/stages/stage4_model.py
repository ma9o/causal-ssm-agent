"""Stage 4: Model Specification & Prior Elicitation.

Orchestrator-Worker architecture with PyMC grounding:
1. Orchestrator proposes GLMM specification
2. Workers research priors in parallel (one per parameter via Exa + LLM)
3. PyMC model is built as grounding step (validates priors compile)
4. Prior predictive checks validate reasonableness

See docs/modeling/functional_spec.md for design rationale.
"""

import polars as pl
from prefect import flow, task


@task(retries=2, retry_delay_seconds=10, task_run_name="propose-glmm-spec")
def propose_glmm_task(
    dsem_model: dict,
    question: str,
    measurements_data: dict[str, pl.DataFrame],
) -> dict:
    """Orchestrator proposes GLMM specification.

    Args:
        dsem_model: Full DSEM model dict
        question: Research question
        measurements_data: Measurement data by granularity

    Returns:
        GLMMSpec as dict
    """
    import asyncio

    from inspect_ai.model import get_model

    from dsem_agent.orchestrator.stage4_orchestrator import (
        build_data_summary,
        propose_glmm_spec,
    )
    from dsem_agent.utils.config import get_config
    from dsem_agent.utils.llm import make_orchestrator_generate_fn

    async def run():
        config = get_config()
        model = get_model(config.stage4_prior_elicitation.model)
        generate = make_orchestrator_generate_fn(model)

        data_summary = build_data_summary(measurements_data)

        result = await propose_glmm_spec(
            dsem_model=dsem_model,
            data_summary=data_summary,
            question=question,
            generate=generate,
        )

        return result.glmm_spec.model_dump()

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

    from dsem_agent.orchestrator.schemas_glmm import ParameterSpec
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
    glmm_spec: dict,
    priors: dict[str, dict],
    measurements_data: dict[str, pl.DataFrame],
) -> dict:
    """Validate priors via prior predictive sampling.

    Args:
        glmm_spec: GLMM specification dict
        priors: Prior proposals by parameter name
        measurements_data: Measurement data

    Returns:
        Validation result dict with is_valid and issues
    """
    from dsem_agent.models.prior_predictive import validate_prior_predictive

    is_valid, results = validate_prior_predictive(
        glmm_spec=glmm_spec,
        priors=priors,
        measurements_data=measurements_data,
    )

    return {
        "is_valid": is_valid,
        "results": [r.model_dump() for r in results],
        "issues": [r.model_dump() for r in results if not r.is_valid],
    }


@task(task_run_name="build-dsem-model")
def build_model_task(
    glmm_spec: dict,
    priors: dict[str, dict],
    measurements_data: dict[str, pl.DataFrame],
) -> dict:
    """Build DSEMModelBuilder from spec and priors.

    Args:
        glmm_spec: GLMM specification
        priors: Prior proposals
        measurements_data: Measurement data

    Returns:
        Dict with model_built status and builder info
    """
    import pandas as pd

    from dsem_agent.models.dsem_model_builder import DSEMModelBuilder

    try:
        builder = DSEMModelBuilder(glmm_spec=glmm_spec, priors=priors)

        dfs = []
        for granularity, df in measurements_data.items():
            if granularity != "time_invariant" and df.height > 0:
                dfs.append(df.to_pandas())

        X = dfs[0] if dfs else pd.DataFrame({"x": [0.0]})

        builder.build_model(X)

        return {
            "model_built": True,
            "model_type": builder._model_type,
            "version": builder.version,
            "output_var": builder.output_var,
        }

    except Exception as e:
        return {
            "model_built": False,
            "error": str(e),
        }


@flow(name="stage4-orchestrated", log_prints=True)
def stage4_orchestrated_flow(
    dsem_model: dict,
    question: str,
    measurements_data: dict[str, pl.DataFrame],
    enable_literature: bool = True,
) -> dict:
    """Stage 4 orchestrated flow with parallel worker prior research.

    1. Orchestrator proposes GLMM specification
    2. Workers research priors in parallel (one per parameter)
    3. Validate via prior predictive checks
    4. Build PyMC model

    Args:
        dsem_model: Full DSEM model dict
        question: Research question
        measurements_data: Measurement data by granularity
        enable_literature: Whether to search Exa for literature

    Returns:
        Stage 4 result dict with glmm_spec, priors, validation
    """
    from prefect.utilities.annotations import unmapped

    from dsem_agent.utils.config import get_config

    config = get_config()
    paraphrasing = config.stage4_prior_elicitation.paraphrasing

    # Determine paraphrasing settings
    n_paraphrases = paraphrasing.n_paraphrases if paraphrasing.enabled else 1

    # 1. Orchestrator proposes GLMM specification
    glmm_spec = propose_glmm_task(dsem_model, question, measurements_data)

    # 2. Workers research priors in parallel
    parameter_specs = glmm_spec.get("parameters", [])
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
        priors[param_name] = prior_result.result() if hasattr(prior_result, "result") else prior_result

    # 3. Validate priors
    validation = validate_priors_task(glmm_spec, priors, measurements_data)
    validation_result = validation.result() if hasattr(validation, "result") else validation

    # 4. Build PyMC model
    model_info = build_model_task(glmm_spec, priors, measurements_data)
    model_result = model_info.result() if hasattr(model_info, "result") else model_info

    return {
        "glmm_spec": glmm_spec,
        "priors": priors,
        "validation": validation_result,
        "model_info": model_result,
        "is_valid": validation_result.get("is_valid", False),
    }
