"""Stage 5: Bayesian inference and intervention analysis.

Fits the SSM model and runs counterfactual interventions to
estimate treatment effects, ranked by effect size.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import polars as pl
from prefect import task

from causal_ssm_agent.models.ssm.inference import FittedArtifact
from causal_ssm_agent.utils.data import pivot_to_wide

from .. import get_prefect_logger

logger = get_prefect_logger(__name__)


@dataclass
class PreparedModelRuntime:
    """Canonical prepared runtime context shared by pre-fit and post-fit diagnostics.

    Bundles the builder, wide-format data, observation arrays, times, and
    manifest metadata. Avoids repeated pivot_to_wide + array extraction
    across stage 4b, 5a, and 5b.
    """

    builder: Any  # SSMModelBuilder
    wide_data: pl.DataFrame
    observations: jnp.ndarray  # (T, n_manifest)
    times: jnp.ndarray  # (T,)
    manifest_names: list[str]


def prepare_model_runtime(
    raw_data: pl.DataFrame,
    compiled_ssm: dict | None = None,
    sampler_config: dict | None = None,
    builder: Any = None,
) -> PreparedModelRuntime:
    """Build or reuse a builder, pivot data, and extract arrays.

    This is the single canonical entry point for model preparation.
    Used by stage 4b (parametric ID), stage 5a (SVI preflight),
    and stage 5b (full inference + diagnostics).

    Args:
        raw_data: Raw timestamped data (indicator, value, timestamp)
        compiled_ssm: Serialized executable artifact from stage 4
        sampler_config: Override sampler configuration
        builder: Pre-built SSMModelBuilder (avoids rebuilding)

    Returns:
        PreparedModelRuntime with all arrays extracted
    """
    from causal_ssm_agent.models.ssm_builder import build_ssm_builder

    wide_data = pivot_to_wide(raw_data)

    if builder is None:
        if compiled_ssm is None:
            raise ValueError("Either builder or compiled_ssm must be provided")
        builder = build_ssm_builder(
            wide_data=wide_data,
            sampler_config=sampler_config,
            compiled_ssm=compiled_ssm,
        )

    observations, times, manifest_names = builder.prepare_fit_inputs(wide_data)

    return PreparedModelRuntime(
        builder=builder,
        wide_data=wide_data,
        observations=observations,
        times=times,
        manifest_names=manifest_names,
    )


@task(persist_result=False)
def fit_model(
    stage4_result: dict,
    raw_data: pl.DataFrame,
    sampler_config: dict | None = None,
    builder: Any = None,
) -> Any:
    """Fit the SSM model to data.

    Args:
        stage4_result: Result from stage4_orchestrated_flow containing
            model_spec, priors, and model_info
        raw_data: Raw timestamped data (indicator, value, timestamp)
        sampler_config: Override sampler configuration (None uses config defaults)
        builder: Pre-built SSMModelBuilder (avoids rebuilding)

    Returns:
        Fitted model results

    NOTE: Uses NumPyro SSM implementation.
    """
    model_spec = stage4_result.get("model_spec", {})
    compiled_ssm = stage4_result.get("_compiled_ssm")
    logger.info(
        "Fitting model: rows=%d indicators=%d parameters=%d sampler=%s builder_reused=%s",
        len(raw_data),
        raw_data["indicator"].n_unique() if "indicator" in raw_data.columns else 0,
        len(model_spec.get("parameters", [])),
        (sampler_config or {}).get("method", "config default"),
        builder is not None,
    )

    try:
        runtime = prepare_model_runtime(
            raw_data=raw_data,
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

        return {
            "fitted": True,
            "inference_type": result.method,
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
        }
    except Exception as e:
        logger.exception("Model fitting failed")
        return {
            "fitted": False,
            "error": str(e),
        }


@task(task_run_name="power-scaling-sensitivity", result_serializer="json")
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


@task(task_run_name="posterior-predictive-checks", result_serializer="json")
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
            manifest_dist=spec.manifest_dist.value
            if hasattr(spec.manifest_dist, "value")
            else str(spec.manifest_dist),
            manifest_dists=spec.manifest_dists,
            manifest_links=spec.manifest_links,
            manifest_level_counts=spec.manifest_level_counts,
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
    ppc_result: dict | None = None,
    ps_result: dict | None = None,
) -> list[dict]:
    """Run do-operator interventions and rank treatments by effect size.

    For each treatment, applies do(treatment = baseline + 1) and measures
    the change in the outcome variable at steady state.

    Args:
        fitted_artifact: Persisted fitted inference artifact
        treatments: List of treatment construct names
        outcome: Name of the outcome variable
        causal_spec: Optional CausalSpec with identifiability status
        ppc_result: Optional PPC result dict for per-treatment warnings

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
        id_status = causal_spec.get("identifiability") if causal_spec else None
        non_identifiable: set[str] = set()
        if id_status:
            non_identifiable = set(id_status.get("non_identifiable_treatments", {}).keys())
        return [
            {
                "treatment": t,
                "effect_size": None,
                "identifiable": t not in non_identifiable,
            }
            for t in treatments
        ]

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
        ppc_result=ppc_result,
        manifest_names=manifest_names,
        ps_result=ps_result,
        times=fitted_artifact.times,
    )
    logger.info("Interventions complete: ranked_treatments=%d", len(results))
    return results
