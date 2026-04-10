"""Stage 5b: Bayesian inference and diagnostics."""

import time
from typing import Any

import jax.numpy as jnp
import polars as pl
from prefect import task

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm_builder import PreparedModelRuntime, prepare_model_runtime

logger = get_prefect_logger(__name__)


def _elapsed_seconds(start: float) -> float:
    return time.monotonic() - start


def _format_name_preview(names: list[str], limit: int = 4) -> str:
    if not names:
        return "none"
    preview = ", ".join(names[:limit])
    if len(names) > limit:
        preview = f"{preview}, ..."
    return preview


def _observed_cell_counts(observations: jnp.ndarray) -> tuple[int, int]:
    total_cells = int(observations.size)
    observed_cells = int(jnp.sum(~jnp.isnan(observations)).item())
    return observed_cells, total_cells


def _time_span_days(times: jnp.ndarray) -> float:
    if times.size <= 1:
        return 0.0
    return float((times[-1] - times[0]).item())


def _support_summary(runtime: PreparedModelRuntime) -> str:
    support = runtime.observation_support
    if support is None:
        return "none"
    if not support.requires_interval_summary_handling:
        return "point-only"
    return (
        f"interval({len(support.interval_summary_manifest_names)}: "
        f"{_format_name_preview(support.interval_summary_manifest_names)}) "
        f"max_active_windows={support.max_active_windows}"
    )


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
        prep_t0 = time.monotonic()
        runtime = prepare_model_runtime(
            data_for_model=data_for_model,
            compiled_ssm=compiled_ssm,
            sampler_config=sampler_config,
            builder=builder,
        )
        observed_cells, total_cells = _observed_cell_counts(runtime.observations)
        logger.info(
            "Prepared runtime in %.1fs: wide_rows=%d timepoints=%d manifest_vars=%d "
            "observed_cells=%d/%d time_span_days=%.2f support=%s",
            _elapsed_seconds(prep_t0),
            len(runtime.wide_data),
            len(runtime.times),
            len(runtime.manifest_names),
            observed_cells,
            total_cells,
            _time_span_days(runtime.times),
            _support_summary(runtime),
        )
        logger.info("Manifest order: %s", _format_name_preview(runtime.manifest_names, limit=6))

        inference_structure = runtime.inference_structure
        logger.info(
            "Inference route: requested_method=%s auto_method=%s likelihood_path=%s "
            "first_pass_rb=%s inactive_reason=%s",
            (sampler_config or {}).get("method", "config default"),
            inference_structure.auto_method,
            inference_structure.likelihood_path,
            inference_structure.first_pass_rb.status,
            inference_structure.first_pass_rb.inactive_reason or "none",
        )
        partition = inference_structure.first_pass_rb.partition
        if partition is not None:
            logger.info(
                "First-pass RB partition: latent_kalman=%d latent_particle=%d "
                "obs_kalman=%d obs_particle=%d",
                len(partition.kalman_idx),
                len(partition.particle_idx),
                len(partition.obs_kalman_idx),
                len(runtime.manifest_names) - len(partition.obs_kalman_idx),
            )

        # Fit the model — returns InferenceResult (default: SVI)
        logger.info("Starting inference kernel...")
        fit_t0 = time.monotonic()
        result = runtime.builder.fit_prepared(runtime.observations, runtime.times)
        logger.info(
            "Inference kernel complete in %.1fs: method=%s wide_rows=%d manifest_vars=%d",
            _elapsed_seconds(fit_t0),
            result.method,
            len(runtime.wide_data),
            len(runtime.manifest_names),
        )

        # Extract serializable diagnostics (MCMC, SVI, or SMC)
        logger.info("Collecting sampler diagnostics...")
        mcmc_diag = result.get_mcmc_diagnostics()
        svi_diag = result.get_svi_diagnostics()
        smc_diag = result.get_smc_diagnostics()

        # LOO diagnostics (needs model function and data).
        # Use the inference-consistent likelihood backend: Laplace-based
        # methods store their backend in diagnostics; MCMC methods fall back
        # to the model's default backend.
        import functools

        assert runtime.builder._model is not None
        loo_backend = result.diagnostics.get(
            "likelihood_backend", runtime.builder._model.make_likelihood_backend()
        )
        model_fn = functools.partial(
            runtime.builder._model.model,
            likelihood_backend=loo_backend,
        )
        logger.info("Computing LOO diagnostics...")
        loo_diag = result.get_loo_diagnostics(
            model_fn=model_fn,
            observations=runtime.observations,
            times=runtime.times,
        )

        # Posterior marginals and pairs
        logger.info("Extracting posterior summaries...")
        posterior_marginals = result.get_posterior_marginals()
        posterior_pairs = result.get_posterior_pairs()
        samples = result.get_samples()
        n_samples = (
            int(next(iter(samples.values())).shape[0])
            if isinstance(samples, dict) and samples
            else 0
        )
        logger.info(
            "Posterior summaries ready in %.1fs: n_samples=%d",
            _elapsed_seconds(t0),
            n_samples,
        )

        return {
            "fitted": True,
            "inference_type": result.method,
            "n_samples": n_samples,
            "duration_seconds": _elapsed_seconds(t0),
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
            "duration_seconds": _elapsed_seconds(t0),
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
    from causal_ssm_agent.models.ssm.diagnostics import power_scaling_sensitivity

    if not fitted_result.get("fitted", False):
        return {"checked": False, "error": "Model not fitted"}

    t0 = time.monotonic()
    try:
        result = fitted_result["result"]
        runtime: PreparedModelRuntime = fitted_result["runtime"]
        assert runtime.builder._model is not None
        ssm_model = runtime.builder._model
        logger.info(
            "Running power-scaling sensitivity: method=%s timepoints=%d manifest_vars=%d",
            result.method,
            len(runtime.times),
            len(runtime.manifest_names),
        )

        ps_result = power_scaling_sensitivity(
            model=ssm_model,
            observations=runtime.observations,
            times=runtime.times,
            result=result,
        )

        ps_result.print_report()
        diagnosis = ps_result.diagnosis or {}
        flagged = sum(
            verdict in {"prior_dominated", "prior_data_conflict"} for verdict in diagnosis.values()
        )
        logger.info(
            "Power-scaling complete in %.1fs: parameters=%d flagged=%d",
            _elapsed_seconds(t0),
            len(diagnosis),
            flagged,
        )

        return {
            "checked": True,
            "prior_sensitivity": ps_result.prior_sensitivity,
            "likelihood_sensitivity": ps_result.likelihood_sensitivity,
            "diagnosis": ps_result.diagnosis,
            "psis_k_hat": ps_result.psis_k_hat,
        }

    except (ValueError, RuntimeError, ArithmeticError, FloatingPointError) as e:
        logger.exception("Power-scaling check failed after %.1fs", _elapsed_seconds(t0))
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

    t0 = time.monotonic()
    try:
        result = fitted_result["result"]
        runtime: PreparedModelRuntime = fitted_result["runtime"]
        spec = runtime.builder._spec
        samples = result.get_samples()
        posterior_draws = (
            int(next(iter(samples.values())).shape[0])
            if isinstance(samples, dict) and samples
            else 0
        )
        logger.info(
            "Running posterior predictive checks: method=%s posterior_draws=%d "
            "timepoints=%d manifest_vars=%d",
            result.method,
            posterior_draws,
            len(runtime.times),
            len(runtime.manifest_names),
        )

        ppc_result = run_posterior_predictive_checks(
            samples=samples,
            observations=runtime.observations,
            times=runtime.times,
            manifest_names=runtime.manifest_names,
            diffusion_dists=spec.diffusion_dists,
            manifest_dists=spec.manifest_dists,
            manifest_links=spec.manifest_links,
            manifest_level_counts=spec.manifest_level_counts,
            observation_support=runtime.observation_support,
            observation_mask=~jnp.isnan(runtime.observations),
        )
        logger.info(
            "Posterior predictive checks complete in %.1fs: warnings=%d",
            _elapsed_seconds(t0),
            len(ppc_result.per_variable_warnings),
        )

        return ppc_result.model_dump(mode="json")

    except (ValueError, RuntimeError, ArithmeticError, FloatingPointError):
        logger.exception("PPC check failed after %.1fs", _elapsed_seconds(t0))
        return {"checked": False, "per_variable_warnings": []}
