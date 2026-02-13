"""Stage 5: Bayesian inference and intervention analysis.

Fits the SSM model and runs counterfactual interventions to
estimate treatment effects, ranked by effect size.
"""

import logging
from typing import Any

import polars as pl
from prefect import task

from causal_ssm_agent.utils.data import pivot_to_wide

logger = logging.getLogger(__name__)


@task(persist_result=False)
def fit_model(
    stage4_result: dict,
    raw_data: pl.DataFrame,
    sampler_config: dict | None = None,
) -> Any:
    """Fit the SSM model to data.

    Args:
        stage4_result: Result from stage4_orchestrated_flow containing
            model_spec, priors, and model_info
        raw_data: Raw timestamped data (indicator, value, timestamp)
        sampler_config: Override sampler configuration (None uses config defaults)

    Returns:
        Fitted model results

    NOTE: Uses NumPyro SSM implementation.
    """
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder

    model_spec = stage4_result.get("model_spec", {})
    priors = stage4_result.get("priors", {})

    try:
        builder = SSMModelBuilder(
            model_spec=model_spec, priors=priors, sampler_config=sampler_config
        )

        # Convert data to wide format
        if raw_data.is_empty():
            return {"fitted": False, "error": "No data available"}

        X = pivot_to_wide(raw_data)

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


@task(task_run_name="power-scaling-sensitivity", result_serializer="json")
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
        X = pivot_to_wide(raw_data)

        observations = jnp.array(X.drop("time").to_numpy(), dtype=jnp.float64)
        times = jnp.array(X["time"].to_numpy(), dtype=jnp.float64)

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
        logger.exception("Power-scaling check failed")
        return {"checked": False, "error": str(e)}


@task(task_run_name="posterior-predictive-checks", result_serializer="json")
def run_ppc(fitted_result: dict, raw_data: pl.DataFrame) -> dict:
    """Run posterior predictive checks on the fitted model.

    Forward-simulates from posterior draws and compares to observed data,
    producing per-variable warnings for calibration, autocorrelation, and variance.

    Args:
        fitted_result: Output from fit_model task
        raw_data: Raw timestamped data (indicator, value, timestamp)

    Returns:
        Dict with PPC diagnostics (PPCResult.to_dict())
    """
    import jax.numpy as jnp

    from causal_ssm_agent.models.posterior_predictive import run_posterior_predictive_checks

    if not fitted_result.get("fitted", False):
        return {"checked": False, "error": "Model not fitted"}

    try:
        result = fitted_result["result"]
        builder = fitted_result["builder"]
        spec = builder._spec
        samples = result.get_samples()

        # Convert data to wide format
        X = pivot_to_wide(raw_data)

        observations = jnp.array(X.drop("time").to_numpy(), dtype=jnp.float64)
        times = jnp.array(X["time"].to_numpy(), dtype=jnp.float64)

        manifest_names = spec.manifest_names or [c for c in X.columns if c != "time"]

        ppc_result = run_posterior_predictive_checks(
            samples=samples,
            observations=observations,
            times=times,
            manifest_names=manifest_names,
            manifest_dist=spec.manifest_dist.value
            if hasattr(spec.manifest_dist, "value")
            else str(spec.manifest_dist),
        )

        return ppc_result.to_dict()

    except Exception as e:
        logger.exception("PPC check failed")
        return {"checked": False, "error": str(e)}


@task(result_serializer="json")
def run_interventions(
    fitted_model: Any,
    treatments: list[str],
    outcome: str,
    causal_spec: dict | None = None,
    ppc_result: dict | None = None,
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
            {
                "treatment": t,
                "effect_size": None,
                "credible_interval": None,
                "identifiable": t not in non_identifiable,
            }
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
        logger.warning("Outcome '%s' not found in latent names %s", outcome, latent_names)
        return [
            {
                "treatment": t,
                "effect_size": None,
                "credible_interval": None,
                "identifiable": t not in non_identifiable,
            }
            for t in treatments
        ]

    # Extract posterior drift and cint draws
    drift_draws = samples.get("drift")  # (n_draws, n, n)
    cint_draws = samples.get("cint")  # (n_draws, n) or None

    if drift_draws is None:
        logger.warning("No 'drift' in posterior samples")
        return [
            {
                "treatment": t,
                "effect_size": None,
                "credible_interval": None,
                "identifiable": t not in non_identifiable,
            }
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

    # Attach PPC warnings to each treatment entry
    if ppc_result and ppc_result.get("checked", False) and ppc_result.get("warnings"):
        from causal_ssm_agent.models.posterior_predictive import get_relevant_manifest_variables

        lambda_mat = samples.get("lambda")
        # Use mean lambda if per-draw
        if lambda_mat is not None and lambda_mat.ndim == 3:
            lambda_mat = jnp.mean(lambda_mat, axis=0)

        manifest_names = spec.manifest_names or []

        for entry in results:
            treat_idx = name_to_idx.get(entry["treatment"])
            relevant_vars = get_relevant_manifest_variables(
                lambda_mat, treat_idx, outcome_idx, manifest_names
            )
            entry_warnings = [
                w for w in ppc_result["warnings"] if w.get("variable") in relevant_vars
            ]
            if entry_warnings:
                entry["ppc_warnings"] = entry_warnings

    return results
