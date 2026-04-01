"""Stage 5: Bayesian inference and intervention analysis.

Fits the SSM model and runs counterfactual interventions to
estimate treatment effects, ranked by effect size.
"""

import time
from typing import Any

import jax.numpy as jnp
import polars as pl
from prefect import task

from causal_ssm_agent.models.ssm.inference import FittedArtifact
from causal_ssm_agent.models.ssm_builder import PreparedModelRuntime, prepare_model_runtime

from .. import get_prefect_logger

logger = get_prefect_logger(__name__)


@task(persist_result=False)
def fit_model(
    compiled_ssm: dict | None,
    data_for_model: pl.DataFrame,
    sampler_config: dict | None = None,
    builder: Any = None,
) -> Any:
    """Fit the SSM model to data.

    Args:
        compiled_ssm: Serialized executable SSM artifact from stage 4
        data_for_model: Canonical observation rows (indicator, value, anchor_time, support metadata)
        sampler_config: Override sampler configuration (None uses config defaults)
        builder: Pre-built SSMModelBuilder (avoids rebuilding)

    Returns:
        Fitted model results

    NOTE: Uses NumPyro SSM implementation.
    """
    logger.info(
        "Fitting model: rows=%d indicators=%d sampler=%s builder_reused=%s",
        len(data_for_model),
        data_for_model["indicator"].n_unique() if "indicator" in data_for_model.columns else 0,
        (sampler_config or {}).get("method", "config default"),
        builder is not None,
    )
    t0 = time.monotonic()

    try:
        runtime = prepare_model_runtime(
            data_for_model=data_for_model,
            compiled_ssm=compiled_ssm,
            sampler_config=sampler_config,
            builder=builder,
        )

        # Fit the model — returns InferenceResult (default: SVI)
        result = runtime.builder.fit_prepared(runtime.observations, runtime.times)
        logger.info(
            "Fit complete: method=%s wide_rows=%d manifest_vars=%d",
            result.method,
            len(runtime.wide_data),
            len(runtime.manifest_names),
        )

        # Extract serializable diagnostics (MCMC, SVI, or SMC)
        mcmc_diag = result.get_mcmc_diagnostics()
        svi_diag = result.get_svi_diagnostics()
        smc_diag = result.get_smc_diagnostics()

        # LOO diagnostics (needs model function and data).
        # Use the inference-consistent likelihood backend: for laplace_em
        # the Laplace backend is stored in diagnostics; for MCMC methods
        # fall back to the model's default backend.
        import functools

        assert runtime.builder._model is not None
        loo_backend = result.diagnostics.get(
            "likelihood_backend", runtime.builder._model.make_likelihood_backend()
        )
        model_fn = functools.partial(
            runtime.builder._model.model,
            likelihood_backend=loo_backend,
        )
        loo_diag = result.get_loo_diagnostics(
            model_fn=model_fn,
            observations=runtime.observations,
            times=runtime.times,
        )

        # Posterior marginals and pairs
        posterior_marginals = result.get_posterior_marginals()
        posterior_pairs = result.get_posterior_pairs()
        samples = result.get_samples()
        n_samples = (
            int(next(iter(samples.values())).shape[0])
            if isinstance(samples, dict) and samples
            else 0
        )

        return {
            "fitted": True,
            "inference_type": result.method,
            "n_samples": n_samples,
            "duration_seconds": time.monotonic() - t0,
            "result": result,
            "builder": runtime.builder,
            "runtime": runtime,
            "times": runtime.times,
            "mcmc_diagnostics": mcmc_diag,
            "svi_diagnostics": svi_diag,
            "smc_diagnostics": smc_diag,
            "loo_diagnostics": loo_diag,
            "posterior_marginals": posterior_marginals,
            "posterior_pairs": posterior_pairs,
        }

    except NotImplementedError:
        logger.warning("SSM implementation not available for model fitting")
        return {
            "fitted": False,
            "error": "SSM implementation not available",
            "duration_seconds": time.monotonic() - t0,
        }
    except Exception as e:
        logger.exception("Model fitting failed")
        return {
            "fitted": False,
            "error": str(e),
            "duration_seconds": time.monotonic() - t0,
        }


@task(task_run_name="power-scaling-sensitivity", persist_result=False)
def run_power_scaling(fitted_result: dict) -> dict:
    """Post-fit power-scaling sensitivity diagnostic.

    Detects prior-dominated, well-identified, or conflicting parameters
    by perturbing prior/likelihood contributions and measuring posterior shift.

    Args:
        fitted_result: Output from fit_model task (includes runtime)

    Returns:
        Dict with power-scaling diagnostics
    """
    from causal_ssm_agent.utils.parametric_id_postfit import power_scaling_sensitivity

    if not fitted_result.get("fitted", False):
        return {"checked": False, "error": "Model not fitted"}

    try:
        result = fitted_result["result"]
        runtime: PreparedModelRuntime = fitted_result["runtime"]
        assert runtime.builder._model is not None
        ssm_model = runtime.builder._model

        ps_result = power_scaling_sensitivity(
            model=ssm_model,
            observations=runtime.observations,
            times=runtime.times,
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


@task(task_run_name="posterior-predictive-checks", persist_result=False)
def run_ppc(fitted_result: dict) -> dict:
    """Run posterior predictive checks on the fitted model.

    Forward-simulates from posterior draws and compares to observed data,
    producing per-variable warnings for calibration, autocorrelation, and variance.

    Args:
        fitted_result: Output from fit_model task (includes runtime)

    Returns:
        Dict with PPC diagnostics (PPCResult.model_dump())
    """
    from causal_ssm_agent.models.posterior_predictive import run_posterior_predictive_checks

    if not fitted_result.get("fitted", False):
        return {"checked": False, "per_variable_warnings": []}

    try:
        result = fitted_result["result"]
        runtime: PreparedModelRuntime = fitted_result["runtime"]
        spec = runtime.builder._spec
        samples = result.get_samples()

        ppc_result = run_posterior_predictive_checks(
            samples=samples,
            observations=runtime.observations,
            times=runtime.times,
            manifest_names=runtime.manifest_names,
            diffusion_dist=spec.diffusion_dist,
            diffusion_dists=spec.diffusion_dists,
            manifest_dist=spec.manifest_dist.value
            if hasattr(spec.manifest_dist, "value")
            else str(spec.manifest_dist),
            manifest_dists=spec.manifest_dists,
            manifest_links=spec.manifest_links,
            manifest_level_counts=spec.manifest_level_counts,
            observation_support=runtime.observation_support,
            observation_mask=~jnp.isnan(runtime.observations),
        )

        return ppc_result.model_dump(mode="json")

    except Exception:
        logger.exception("PPC check failed")
        return {"checked": False, "per_variable_warnings": []}


@task(result_serializer="json")
def run_interventions(
    fitted_artifact: FittedArtifact,
    treatments: list[str],
    outcome: str,
    causal_spec: dict | None = None,
) -> list[dict]:
    """Run do-operator interventions and rank treatments by effect size.

    For each treatment, applies do(treatment = baseline + 1) and measures
    the change in the outcome variable at steady state.

    Args:
        fitted_artifact: Persisted fitted inference artifact
        treatments: List of treatment construct names
        outcome: Name of the outcome variable
        causal_spec: Optional CausalSpec with identifiability status

    Returns:
        List of intervention results, sorted by |effect_size| descending
    """
    from causal_ssm_agent.models.ssm.counterfactual import compute_interventions

    logger.info(
        "Running interventions: treatments=%d outcome=%s fitted=%s",
        len(treatments),
        outcome or "unknown",
        fitted_artifact.result is not None,
    )

    # If model not fitted, return skeleton results
    if fitted_artifact.result is None or fitted_artifact.builder is None:
        return [{"treatment": t} for t in treatments]

    builder = fitted_artifact.builder
    result = fitted_artifact.result
    samples = result.get_samples()
    spec = builder._spec

    latent_names = spec.latent_names
    if latent_names is None:
        latent_names = spec.manifest_names or []

    manifest_names = spec.manifest_names or []

    results = compute_interventions(
        samples=samples,
        treatments=treatments,
        outcome=outcome,
        latent_names=latent_names,
        causal_spec=causal_spec,
        manifest_names=manifest_names,
        times=fitted_artifact.times,
    )
    logger.info("Interventions complete: ranked_treatments=%d", len(results))
    return results
