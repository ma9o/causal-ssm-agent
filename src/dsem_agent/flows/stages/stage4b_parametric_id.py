"""Stage 4b: Parametric Identifiability Diagnostics.

Pre-fit diagnostics that check whether model parameters are constrained
by the data before running expensive inference. Sits between Stage 4
(model specification) and Stage 5 (inference).

Detects:
- Structural non-identifiability (rank-deficient Fisher information)
- Boundary identifiability (intermittent rank deficiency)
- Weak parameters (low expected prior→posterior contraction)
- Uninformative estimands (target quantities not constrained by data)
"""

import polars as pl
from prefect import flow, task


@task(task_run_name="parametric-id-check")
def parametric_id_task(
    model_spec: dict,
    priors: dict[str, dict],
    raw_data: pl.DataFrame,
    n_draws: int = 5,
) -> dict:
    """Run parametric identifiability checks.

    1. Build SSMModel from spec + priors
    2. Prepare data (pivot raw → wide)
    3. Call check_parametric_id()
    4. Return result summary

    Args:
        model_spec: Model specification dict
        priors: Prior proposals by parameter name
        raw_data: Raw timestamped data (indicator, value, timestamp)
        n_draws: Number of prior draws for Hessian analysis

    Returns:
        Dict with parametric ID diagnostics
    """
    import jax.numpy as jnp

    from dsem_agent.models.ssm_builder import SSMModelBuilder
    from dsem_agent.utils.parametric_id import check_parametric_id

    try:
        builder = SSMModelBuilder(model_spec=model_spec, priors=priors)

        if raw_data.is_empty():
            return {"checked": False, "error": "No data available"}

        # Pivot raw data to wide format
        wide_data = (
            raw_data.with_columns(pl.col("value").cast(pl.Float64, strict=False))
            .pivot(on="indicator", index="timestamp", values="value")
            .sort("timestamp")
        )

        X = wide_data.to_pandas()
        if "timestamp" in X.columns:
            X = X.rename(columns={"timestamp": "time"})

        # Build the model
        builder.build_model(X)
        ssm_model = builder._model

        # Extract observations and times
        observations = jnp.array(X.drop(columns=["time"]).values, dtype=jnp.float32)
        times = jnp.array(X["time"].values, dtype=jnp.float32)

        # Run parametric ID check
        result = check_parametric_id(
            model=ssm_model,
            observations=observations,
            times=times,
            n_draws=n_draws,
        )

        result.print_report()
        summary = result.summary()

        return {
            "checked": True,
            "summary": summary,
            "n_draws": n_draws,
            "n_parameters": len(result.parameter_names),
            "parameter_names": result.parameter_names,
        }

    except Exception as e:
        print(f"Parametric ID check failed: {e}")
        return {
            "checked": False,
            "error": str(e),
        }


@flow(name="stage4b-parametric-id", log_prints=True)
def stage4b_parametric_id_flow(
    stage4_result: dict,
) -> dict:
    """Stage 4b: Parametric identifiability check.

    Takes stage4 output, runs pre-fit diagnostics,
    returns augmented result with parametric ID info.

    Args:
        stage4_result: Output from stage4_orchestrated_flow

    Returns:
        stage4_result augmented with 'parametric_id' key
    """
    model_spec = stage4_result["model_spec"]
    priors = stage4_result["priors"]
    raw_data = stage4_result["raw_data"]

    id_result = parametric_id_task(model_spec, priors, raw_data)

    return {
        **stage4_result,
        "parametric_id": id_result,
    }
