"""Stage 4b: Parametric Identifiability Diagnostics.

Pre-fit diagnostics that check whether model parameters are constrained
by the data before running expensive inference. Sits between Stage 4
(model specification) and Stage 5 (inference).

Detects:
- Structural non-identifiability (flat profile likelihood)
- Practical non-identifiability (profile doesn't cross threshold)
- Well-identified parameters (profile crosses threshold on both sides)
"""

import polars as pl
from prefect import flow, task


@task(task_run_name="parametric-id-check")
def parametric_id_task(
    model_spec: dict,
    priors: dict[str, dict],
    raw_data: pl.DataFrame,
    n_grid: int = 20,
    confidence: float = 0.95,
) -> dict:
    """Run parametric identifiability checks via profile likelihood.

    1. Build SSMModel from spec + priors
    2. Prepare data (pivot raw -> wide)
    3. Call profile_likelihood()
    4. Return result summary

    Args:
        model_spec: Model specification dict
        priors: Prior proposals by parameter name
        raw_data: Raw timestamped data (indicator, value, timestamp)
        n_grid: Number of grid points for profile likelihood
        confidence: Confidence level for chi-squared threshold

    Returns:
        Dict with parametric ID diagnostics
    """
    import jax.numpy as jnp

    from dsem_agent.models.ssm_builder import SSMModelBuilder
    from dsem_agent.utils.parametric_id import profile_likelihood

    try:
        builder = SSMModelBuilder(model_spec=model_spec, priors=priors)

        if raw_data.is_empty():
            return {"checked": False, "error": "No data available"}

        # Pivot data to wide format
        time_col = "time_bucket" if "time_bucket" in raw_data.columns else "timestamp"
        wide_data = (
            raw_data.with_columns(pl.col("value").cast(pl.Float64, strict=False))
            .pivot(on="indicator", index=time_col, values="value")
            .sort(time_col)
        )

        X = wide_data.to_pandas()
        if time_col in X.columns:
            X = X.rename(columns={time_col: "time"})

        # Build the model
        builder.build_model(X)
        ssm_model = builder._model

        # Extract observations and times
        observations = jnp.array(X.drop(columns=["time"]).values, dtype=jnp.float32)
        times = jnp.array(X["time"].values, dtype=jnp.float32)
        T = int(times.shape[0])

        # T-rule: fast necessary condition (hard gate)
        from dsem_agent.utils.parametric_id import check_t_rule

        t_rule = check_t_rule(ssm_model.spec, T=T)
        t_rule.print_report()

        if not t_rule.satisfies:
            return {
                "checked": True,
                "t_rule": {
                    "satisfies": False,
                    "n_free_params": t_rule.n_free_params,
                    "n_moments": t_rule.n_moments,
                    "param_counts": t_rule.param_counts,
                },
                "summary": {},
                "error": (
                    f"T-rule violated: {t_rule.n_free_params} free params "
                    f"> {t_rule.n_moments} moment conditions. "
                    "Model is provably non-identified."
                ),
            }

        # Run profile likelihood check
        result = profile_likelihood(
            model=ssm_model,
            observations=observations,
            times=times,
            n_grid=n_grid,
            confidence=confidence,
        )

        result.print_report()
        summary = result.summary()

        return {
            "checked": True,
            "t_rule": {
                "satisfies": True,
                "n_free_params": t_rule.n_free_params,
                "n_moments": t_rule.n_moments,
            },
            "summary": summary,
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
