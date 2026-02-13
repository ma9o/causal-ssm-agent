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
    fitted_model: Any,
    treatments: list[str],
    outcome: str,
    causal_spec: dict | None = None,
) -> list[dict]:
    """Run do-operator interventions and rank treatments by effect size.

    For each treatment, applies do(treatment = baseline + 1) and measures
    the change in the outcome variable at steady state.

    Args:
        fitted_model: The fitted model result from fit_model
        treatments: List of treatment construct names
        outcome: Name of the outcome variable
        causal_spec: Optional CausalSpec with identifiability status

    Returns:
        List of intervention results, sorted by |effect_size| descending
    """
    import jax.numpy as jnp
    from jax import vmap

    from causal_ssm_agent.models.ssm.counterfactual import treatment_effect

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

    # If model not fitted, return skeleton results
    if not fitted_model.get("fitted", False):
        return [
            {"treatment": t, "effect_size": None, "credible_interval": None, "identifiable": True}
            for t in treatments
        ]

    builder = fitted_model["builder"]
    result = fitted_model["result"]
    samples = result.get_samples()

    # Resolve latent names → drift indices
    spec = builder._spec
    latent_names = spec.latent_names
    if latent_names is None:
        # Fallback: use manifest names (identity lambda)
        latent_names = spec.manifest_names or []

    name_to_idx = {name: i for i, name in enumerate(latent_names)}

    outcome_idx = name_to_idx.get(outcome)
    if outcome_idx is None:
        print(f"Warning: outcome '{outcome}' not found in latent names {latent_names}")
        return [
            {"treatment": t, "effect_size": None, "credible_interval": None, "identifiable": True}
            for t in treatments
        ]

    # Extract posterior drift and cint draws
    drift_draws = samples.get("drift")  # (n_draws, n, n)
    cint_draws = samples.get("cint")  # (n_draws, n) or None

    if drift_draws is None:
        print("Warning: no 'drift' in posterior samples")
        return [
            {"treatment": t, "effect_size": None, "credible_interval": None, "identifiable": True}
            for t in treatments
        ]

    # Default cint to zeros if not present
    n_latent = drift_draws.shape[-1]
    if cint_draws is None:
        cint_draws = jnp.zeros((drift_draws.shape[0], n_latent))

    results = []
    for treatment_name in treatments:
        treat_idx = name_to_idx.get(treatment_name)
        if treat_idx is None:
            results.append(
                {
                    "treatment": treatment_name,
                    "effect_size": None,
                    "credible_interval": None,
                    "identifiable": treatment_name not in non_identifiable,
                    "warning": f"'{treatment_name}' not in latent model",
                }
            )
            continue

        # Vmap treatment_effect over posterior draws
        effects = vmap(lambda d, c, ti=treat_idx, oi=outcome_idx: treatment_effect(d, c, ti, oi))(
            drift_draws, cint_draws
        )

        mean_effect = float(jnp.mean(effects))
        q025 = float(jnp.percentile(effects, 2.5))
        q975 = float(jnp.percentile(effects, 97.5))
        prob_positive = float(jnp.mean(effects > 0))

        entry = {
            "treatment": treatment_name,
            "effect_size": mean_effect,
            "credible_interval": (q025, q975),
            "prob_positive": prob_positive,
            "identifiable": treatment_name not in non_identifiable,
        }

        if treatment_name in non_identifiable:
            blockers = blocker_details.get(treatment_name, [])
            if blockers:
                entry["warning"] = f"Effect not identifiable (blocked by: {', '.join(blockers)})"
            else:
                entry["warning"] = "Effect not identifiable (missing proxies)"

        results.append(entry)

    # Sort by |effect_size| descending
    results.sort(
        key=lambda x: abs(x["effect_size"]) if x["effect_size"] is not None else 0, reverse=True
    )

    return results
