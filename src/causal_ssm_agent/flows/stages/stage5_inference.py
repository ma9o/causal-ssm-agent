"""Stage 5: Bayesian inference and intervention analysis.

Fits the SSM model and runs counterfactual interventions to
estimate treatment effects, ranked by effect size.
"""

from typing import Any

import polars as pl
from prefect import task


@task
def fit_model(stage4_result: dict, raw_data: pl.DataFrame) -> Any:
    """Fit the SSM model to data.

    Args:
        stage4_result: Result from stage4_orchestrated_flow containing
            model_spec, priors, and model_info
        raw_data: Raw timestamped data (indicator, value, timestamp)

    Returns:
        Fitted model results

    NOTE: Uses NumPyro SSM implementation.
    """
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

    model_spec = stage4_result.get("model_spec", {})
    priors = stage4_result.get("priors", {})

    try:
        builder = SSMModelBuilder(model_spec=model_spec, priors=priors)

        # Convert data to wide format
        if raw_data.is_empty():
            return {"fitted": False, "error": "No data available"}

        time_col = "time_bucket" if "time_bucket" in raw_data.columns else "timestamp"
        wide_data = (
            raw_data.with_columns(pl.col("value").cast(pl.Float64, strict=False))
            .pivot(on="indicator", index=time_col, values="value")
            .sort(time_col)
        )

        X = wide_data.to_pandas()
        if time_col in X.columns:
            X = X.rename(columns={time_col: "time"})

        # Fit the model — returns InferenceResult (default: SVI)
        result = builder.fit(X)

        return {
            "fitted": True,
            "inference_type": result.method,
            "result": result,
            "builder": builder,
        }

    except NotImplementedError:
        return {
            "fitted": False,
            "error": "SSM implementation not available",
        }
    except Exception as e:
        return {
            "fitted": False,
            "error": str(e),
        }


@task(task_run_name="power-scaling-sensitivity")
def run_power_scaling(fitted_result: dict, raw_data: pl.DataFrame) -> dict:
    """Post-fit power-scaling sensitivity diagnostic.

    Detects prior-dominated, well-identified, or conflicting parameters
    by perturbing prior/likelihood contributions and measuring posterior shift.

    Args:
        fitted_result: Output from fit_model task
        raw_data: Raw timestamped data (indicator, value, timestamp)

    Returns:
        Dict with power-scaling diagnostics
    """
    import jax.numpy as jnp

    from causal_ssm_agent.utils.parametric_id import power_scaling_sensitivity

    if not fitted_result.get("fitted", False):
        return {"checked": False, "error": "Model not fitted"}

    try:
        result = fitted_result["result"]
        builder = fitted_result["builder"]
        ssm_model = builder._model

        # Convert data to wide format
        time_col = "time_bucket" if "time_bucket" in raw_data.columns else "timestamp"
        wide_data = (
            raw_data.with_columns(pl.col("value").cast(pl.Float64, strict=False))
            .pivot(on="indicator", index=time_col, values="value")
            .sort(time_col)
        )
        X = wide_data.to_pandas()
        if time_col in X.columns:
            X = X.rename(columns={time_col: "time"})

        observations = jnp.array(X.drop(columns=["time"]).values, dtype=jnp.float32)
        times = jnp.array(X["time"].values, dtype=jnp.float32)

        ps_result = power_scaling_sensitivity(
            model=ssm_model,
            observations=observations,
            times=times,
            result=result,
        )

        ps_result.print_report()

        return {
            "checked": True,
            "prior_sensitivity": ps_result.prior_sensitivity,
            "likelihood_sensitivity": ps_result.likelihood_sensitivity,
            "diagnosis": ps_result.diagnosis,
            "psis_k_hat": ps_result.psis_k_hat,
        }

    except Exception as e:
        print(f"Power-scaling check failed: {e}")
        return {"checked": False, "error": str(e)}


@task
def run_interventions(
    fitted_model: Any,  # noqa: ARG001
    treatments: list[str],
    causal_spec: dict | None = None,
) -> list[dict]:
    """Run interventions and rank treatments by effect size.

    Args:
        fitted_model: The fitted model result from fit_model
        treatments: List of treatment construct names
        causal_spec: Optional CausalSpec with identifiability status

    Returns:
        List of intervention results, sorted by effect size (descending)

    TODO: Implement intervention analysis and ranking.
    """
    results = []

    # Get identifiability status
    id_status = causal_spec.get("identifiability") if causal_spec else None
    non_identifiable: set[str] = set()
    blocker_details: dict[str, list[str]] = {}
    if id_status:
        non_identifiable_map = id_status.get("non_identifiable_treatments", {})
        non_identifiable = set(non_identifiable_map.keys())
        blocker_details = {
            treatment: details.get("confounders", [])
            for treatment, details in non_identifiable_map.items()
            if isinstance(details, dict)
        }

    for treatment in treatments:
        result = {
            "treatment": treatment,
            "effect_size": None,  # TODO: compute from fitted model
            "credible_interval": None,
            "identifiable": treatment not in non_identifiable,
        }

        if treatment in non_identifiable:
            blockers = blocker_details.get(treatment, [])
            if blockers:
                result["warning"] = f"Effect not identifiable (blocked by: {', '.join(blockers)})"
            else:
                result["warning"] = "Effect not identifiable (missing proxies)"

        results.append(result)

    # TODO: Sort by effect size once computed
    # results.sort(key=lambda x: x['effect_size'] or 0, reverse=True)

    return results
