"""Stage 4b: Parametric Identifiability Diagnostics.

Pre-fit diagnostics that check whether model parameters are constrained
by the data before running expensive inference. Sits between Stage 4
(model specification) and Stage 5 (inference).

Detects:
- Structural non-identifiability (flat profile likelihood)
- Practical non-identifiability (profile doesn't cross threshold)
- Well-identified parameters (profile crosses threshold on both sides)
"""

import logging
from typing import Any

import polars as pl
from prefect import flow, task

from causal_ssm_agent.utils.data import pivot_to_wide

logger = logging.getLogger(__name__)


@task(task_run_name="parametric-id-check")
def parametric_id_task(
    model_spec: dict,
    priors: dict[str, dict],
    raw_data: pl.DataFrame,
    n_grid: int = 20,
    confidence: float = 0.95,
    causal_spec: dict | None = None,
    builder: Any = None,
) -> dict:
    """Run parametric identifiability checks via profile likelihood.

    1. Build SSMModel from spec + priors (or reuse provided builder)
    2. Prepare data (pivot raw -> wide)
    3. Call profile_likelihood()
    4. Return result summary

    Args:
        model_spec: Model specification dict
        priors: Prior proposals by parameter name
        raw_data: Raw timestamped data (indicator, value, timestamp)
        n_grid: Number of grid points for profile likelihood
        confidence: Confidence level for chi-squared threshold
        causal_spec: CausalSpec dict for DAG-constrained masks
        builder: Pre-built SSMModelBuilder (avoids rebuilding)

    Returns:
        Dict with parametric ID diagnostics
    """
    import jax.numpy as jnp

    from causal_ssm_agent.models.ssm_builder import build_ssm_builder
    from causal_ssm_agent.utils.parametric_id import profile_likelihood

    try:
        if builder is None:
            builder = build_ssm_builder(
                model_spec=model_spec,
                priors=priors,
                raw_data=raw_data,
                causal_spec=causal_spec,
            )
        assert builder._model is not None
        ssm_model = builder._model

        # Extract observations and times
        X = pivot_to_wide(raw_data)
        observations = jnp.array(X.drop("time").to_numpy(), dtype=jnp.float32)
        times = jnp.array(X["time"].to_numpy(), dtype=jnp.float32)
        T = int(times.shape[0])

        # T-rule: fast necessary condition (hard gate)
        from causal_ssm_agent.utils.parametric_id import check_t_rule

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

        # Compute RB partition to restrict profiling to Kalman-block params
        kalman_indices = None
        if ssm_model.spec.first_pass_rb:
            try:
                from causal_ssm_agent.models.likelihoods.graph_analysis import (
                    analyze_first_pass_rb,
                    kalman_block_profile_indices,
                )

                partition = analyze_first_pass_rb(ssm_model.spec)
                if partition.has_particle_block:
                    kalman_indices = kalman_block_profile_indices(ssm_model.spec, partition)
                    logger.info(
                        "RB partition: profiling %d/%d Kalman-block params (skipping particle block)",
                        len(kalman_indices),
                        ssm_model.spec.n_latent,
                    )
            except Exception:
                logger.debug("RB partition for profile filtering failed", exc_info=True)

        # Run profile likelihood check (only Kalman-block params when mixed)
        result = profile_likelihood(
            model=ssm_model,
            observations=observations,
            times=times,
            profile_indices=kalman_indices,
            n_grid=n_grid,
            confidence=confidence,
        )

        result.print_report()
        summary = result.summary()

        # Build per-parameter classifications with profile curve data
        per_param = []
        for name in result.parameter_names:
            profile = result.parameter_profiles[name]
            classification = summary[name]
            peak_ll = float(jnp.max(profile["profile_ll"]))
            per_param.append(
                {
                    "name": name,
                    "classification": classification,
                    "profile_x": [float(v) for v in profile["grid_con"]],
                    "profile_ll": [float(v) - peak_ll for v in profile["profile_ll"]],
                }
            )

        return {
            "checked": True,
            "t_rule": {
                "satisfies": True,
                "n_free_params": t_rule.n_free_params,
                "n_moments": t_rule.n_moments,
            },
            "summary": summary,
            "per_param_classification": per_param,
            "threshold": float(result.threshold),
            "n_parameters": len(result.parameter_names),
            "parameter_names": result.parameter_names,
        }

    except Exception as e:
        logger.exception("Parametric ID check failed")
        return {
            "checked": False,
            "error": str(e),
        }


def _compute_rb_partition_payload(builder: Any) -> dict | None:
    """Compute RB partition from builder for both profile filtering and UI."""
    try:
        from causal_ssm_agent.models.likelihoods.graph_analysis import analyze_first_pass_rb

        assert builder._model is not None
        spec = builder._model.spec
        partition = analyze_first_pass_rb(spec)
        latent_names = spec.latent_names or [f"latent_{i}" for i in range(spec.n_latent)]
        manifest_names = spec.manifest_names or [f"obs_{i}" for i in range(spec.n_manifest)]

        return {
            "latent_variables": [
                {
                    "name": latent_names[i],
                    "method": "kalman" if i in partition.kalman_idx else "particle",
                }
                for i in range(spec.n_latent)
            ],
            "obs_variables": [
                {
                    "name": manifest_names[i],
                    "method": "kalman" if i in partition.obs_kalman_idx else "particle",
                }
                for i in range(spec.n_manifest)
            ],
        }
    except Exception:
        logger.debug("RB partition computation skipped", exc_info=True)
        return None


@flow(name="stage4b-parametric-id", log_prints=True, persist_result=True, result_serializer="json")
def stage4b_parametric_id_flow(
    stage4_result: dict,
    raw_data: pl.DataFrame,
    builder: Any = None,
) -> dict:
    """Stage 4b: Parametric identifiability check.

    Takes stage4 output, runs pre-fit diagnostics,
    returns augmented result with parametric ID info.

    Args:
        stage4_result: Output from stage4_orchestrated_flow
        raw_data: Raw timestamped data (indicator, value, timestamp)
        builder: Pre-built SSMModelBuilder (avoids rebuilding)

    Returns:
        stage4_result augmented with 'parametric_id' and 'rb_partition' keys
    """
    model_spec = stage4_result["model_spec"]
    priors = stage4_result["priors"]
    causal_spec = stage4_result.get("causal_spec")

    # Compute RB partition once — reused for profile filtering and UI
    rb_partition = _compute_rb_partition_payload(builder) if builder is not None else None

    id_result = parametric_id_task(
        model_spec, priors, raw_data, causal_spec=causal_spec, builder=builder
    )

    return {
        **stage4_result,
        "parametric_id": id_result,
        "rb_partition": rb_partition,
    }
