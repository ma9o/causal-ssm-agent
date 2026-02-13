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
    causal_spec = stage4_result.get("causal_spec")

    try:
        builder = SSMModelBuilder(
            model_spec=model_spec,
            priors=priors,
            sampler_config=sampler_config,
            causal_spec=causal_spec,
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

        observations = jnp.array(X.drop("time").to_numpy(), dtype=jnp.float32)
        times = jnp.array(X["time"].to_numpy(), dtype=jnp.float32)

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

        observations = jnp.array(X.drop("time").to_numpy(), dtype=jnp.float32)
        times = jnp.array(X["time"].to_numpy(), dtype=jnp.float32)

        manifest_names = spec.manifest_names or [c for c in X.columns if c != "time"]

        # Per-channel distributions override scalar fallback
        manifest_dists_list = None
        if spec.manifest_dists:
            manifest_dists_list = [
                d.value if hasattr(d, "value") else str(d) for d in spec.manifest_dists
            ]

        ppc_result = run_posterior_predictive_checks(
            samples=samples,
            observations=observations,
            times=times,
            manifest_names=manifest_names,
            manifest_dist=spec.manifest_dist.value
            if hasattr(spec.manifest_dist, "value")
            else str(spec.manifest_dist),
            manifest_dists=manifest_dists_list,
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
    ps_result: dict | None = None,
) -> list[dict]:
    """Run do-operator interventions and rank treatments by effect size.

    For each treatment, applies do(treatment = baseline + 1) and measures
    the change in the outcome variable at steady state.

    Args:
        fitted_model: The fitted model result from fit_model
        treatments: List of treatment construct names
        outcome: Name of the outcome variable
        causal_spec: Optional CausalSpec with identifiability status
        ppc_result: Optional PPC result dict for per-treatment warnings

    Returns:
        List of intervention results, sorted by |effect_size| descending
    """
    from causal_ssm_agent.models.ssm.counterfactual import compute_interventions

    # If model not fitted, return skeleton results
    if not fitted_model.get("fitted", False):
        id_status = causal_spec.get("identifiability") if causal_spec else None
        non_identifiable: set[str] = set()
        if id_status:
            non_identifiable = set(id_status.get("non_identifiable_treatments", {}).keys())
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
    spec = builder._spec

    latent_names = spec.latent_names
    if latent_names is None:
        latent_names = spec.manifest_names or []

    manifest_names = spec.manifest_names or []

    return compute_interventions(
        samples=samples,
        treatments=treatments,
        outcome=outcome,
        latent_names=latent_names,
        causal_spec=causal_spec,
        ppc_result=ppc_result,
        manifest_names=manifest_names,
        ps_result=ps_result,
    )
