"""Stage 5b: Bayesian inference and diagnostics."""

import logging
import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import polars as pl

from nof1_causal_lab.flows.stage4_compile_cache import restore_stage4_compile_cache
from nof1_causal_lab.models.ssm.runtime import (
    PreparedModelRuntime,
    fit_prepared_model,
    prepare_model_runtime,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference import InferenceResult

logger = logging.getLogger(__name__)


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


def fit_model(
    compiled_ssm: dict | None,
    data_for_model: pl.DataFrame,
    sampler_config: dict | None = None,
    model: Any = None,
    workspace_id: str | None = None,
    wait_for_compile_cache: bool = False,
    compute_loo_diagnostics: bool = True,
) -> Any:
    """Fit the SSM model to data.

    Args:
        compiled_ssm: Serialized executable SSM artifact from stage 4
        data_for_model: Canonical observation rows (indicator, value, anchor_time, support metadata)
        sampler_config: Override sampler configuration (None uses config defaults)
        model: Optional pre-built SSMModel

    Returns:
        Fitted model results

    NOTE: Uses NumPyro SSM implementation.
    """
    logger.info(
        "Fitting model: rows=%d indicators=%d sampler=%s model_provided=%s",
        len(data_for_model),
        data_for_model["indicator"].n_unique() if "indicator" in data_for_model.columns else 0,
        (sampler_config or {}).get("method", "config default"),
        model is not None,
    )
    t0 = time.monotonic()

    cache_restored = restore_stage4_compile_cache(
        workspace_id,
        compiled_ssm,
        wait_for_pending=wait_for_compile_cache,
    )
    logger.info(
        "Compile cache restore: restored=%s wait_for_pending=%s workspace_id=%s",
        cache_restored,
        wait_for_compile_cache,
        workspace_id or "none",
    )

    try:
        prep_t0 = time.monotonic()
        runtime = prepare_model_runtime(
            data_for_model=data_for_model,
            compiled_ssm=compiled_ssm,
            sampler_config=sampler_config,
            model=model,
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
            "Inference route: requested_method=%s resolved_method=%s structural_backend=%s "
            "method_override=%s",
            (sampler_config or {}).get("method", "config default"),
            inference_structure.resolved_method,
            inference_structure.structural_backend,
            inference_structure.method_override or "none",
        )

        # Fit the model — returns InferenceResult.
        logger.info("Starting inference kernel...")
        fit_t0 = time.monotonic()
        result = fit_prepared_model(runtime)
        logger.info(
            "Inference kernel complete in %.1fs: method=%s wide_rows=%d manifest_vars=%d",
            _elapsed_seconds(fit_t0),
            result.method,
            len(runtime.wide_data),
            len(runtime.manifest_names),
        )

        # Extract serializable diagnostics (MCMC or SMC)
        logger.info("Collecting sampler diagnostics...")
        mcmc_diag = result.get_mcmc_diagnostics()
        smc_diag = result.get_smc_diagnostics()

        loo_diag = None
        if compute_loo_diagnostics:
            # LOO diagnostics (needs model function and data).
            import functools

            loo_backend = result.diagnostics["likelihood_backend"]
            model_fn = functools.partial(
                runtime.model.model,
                likelihood_backend=loo_backend,
            )
            logger.info("Computing LOO diagnostics...")
            loo_diag = result.get_loo_diagnostics(
                model_fn=model_fn,
                observations=runtime.observations,
                times=runtime.times,
            )
        else:
            logger.info("Skipping LOO diagnostics by configuration.")

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
            "spec": runtime.spec,
            "runtime": runtime,
            "times": runtime.times,
            "mcmc_diagnostics": mcmc_diag,
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


def run_ppc(fitted_result: dict) -> dict:
    """Run posterior predictive checks on the fitted model.

    Forward-simulates from posterior draws and compares to observed data,
    producing per-variable warnings for calibration, autocorrelation, and variance.

    Args:
        fitted_result: Output from fit_model task (includes runtime)

    Returns:
        Dict with PPC diagnostics (PPCResult.model_dump())
    """
    from nof1_causal_lab.models.posterior_predictive import run_posterior_predictive_checks

    if not fitted_result.get("fitted", False):
        return {"checked": False, "per_variable_warnings": []}

    t0 = time.monotonic()
    try:
        result: InferenceResult = fitted_result["result"]
        runtime: PreparedModelRuntime = fitted_result["runtime"]
        spec = runtime.spec
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
            spec=spec,
            observation_support=runtime.observation_support,
            observation_mask=~jnp.isnan(runtime.observations),
            transition_inputs=getattr(runtime, "transition_inputs", None),
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
