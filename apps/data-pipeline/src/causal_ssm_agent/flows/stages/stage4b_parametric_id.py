"""Stage 4b: Parametric Identifiability Diagnostics.

Pre-fit diagnostics that check whether model parameters are constrained
by the data before running expensive inference. Sits between Stage 4
(model specification) and Stage 5 (inference).

Detects:
- Structural non-identifiability (flat profile likelihood)
- Practical non-identifiability (profile doesn't cross threshold)
- Well-identified parameters (profile crosses threshold on both sides)
"""

from typing import Any

import polars as pl
from prefect import flow, task

from .. import get_prefect_logger

logger = get_prefect_logger(__name__)


@task(task_run_name="parametric-id-check")
def parametric_id_task(
    _model_spec: dict,
    _priors: dict[str, dict],
    raw_data: pl.DataFrame,
    n_grid: int = 20,
    confidence: float = 0.95,
    _causal_spec: dict | None = None,
    compiled_ssm: dict | None = None,
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
        raw_data: Canonical observation rows (indicator, value, anchor_time, support metadata)
        n_grid: Number of grid points for profile likelihood
        confidence: Confidence level for chi-squared threshold
        causal_spec: CausalSpec dict for DAG-constrained masks
        compiled_ssm: Serialized executable artifact from stage 4
        builder: Pre-built SSMModelBuilder (avoids rebuilding)

    Returns:
        Dict with parametric ID diagnostics and the prepared inference structure
    """
    import jax.numpy as jnp

    from causal_ssm_agent.models.ssm.inference_structure import (
        build_inference_structure_payload,
    )
    from causal_ssm_agent.models.ssm_builder import prepare_model_runtime
    from causal_ssm_agent.utils.parametric_id import profile_likelihood

    try:
        runtime = prepare_model_runtime(
            raw_data=raw_data,
            compiled_ssm=compiled_ssm,
            builder=builder,
        )
        assert runtime.builder._model is not None
        ssm_model = runtime.builder._model
        observations = runtime.observations
        times = runtime.times
        T = int(times.shape[0])

        inference_structure_payload = build_inference_structure_payload(
            ssm_model.spec,
            runtime.inference_structure,
        )

        # T-rule: fast necessary condition (hard gate)
        from causal_ssm_agent.utils.parametric_id import check_t_rule

        t_rule = check_t_rule(ssm_model.spec, T=T)
        t_rule.print_report()

        if not t_rule.satisfies:
            return {
                "parametric_id": {
                    "checked": True,
                    "t_rule": t_rule.model_dump(),
                    "summary": {},
                    "error": (
                        f"T-rule violated: {t_rule.n_free_params} free params "
                        f"> {t_rule.n_moments} moment conditions. "
                        "Model is provably non-identified."
                    ),
                },
                "inference_structure": inference_structure_payload,
            }

        from causal_ssm_agent.utils.parametric_id import get_stage4b_sweep_context

        sweep_context = get_stage4b_sweep_context(ssm_model)

        # Sensitivity analysis: structural check (sufficient for local identifiability)
        sensitivity_payload = None
        try:
            from causal_ssm_agent.utils.parametric_id import output_sensitivity_analysis

            sa_result = output_sensitivity_analysis(
                ssm_model,
                times,
                n_draws=8,
                seed=42,
                sweep_context=sweep_context,
            )
            sa_result.print_report()
            sensitivity_payload = {
                "singular_values": sa_result.singular_values,
                "condition_number": sa_result.condition_number,
                "per_parameter": sa_result.per_parameter,
                "n_draws": sa_result.n_draws,
                "n_observations": sa_result.n_observations,
                "n_parameters": sa_result.n_parameters,
            }
        except Exception:
            logger.debug(
                "Sensitivity analysis failed, continuing with profile likelihood", exc_info=True
            )

        # Restrict profiling to the active first-pass Kalman block when a
        # composed likelihood path is actually available in the prepared runtime.
        kalman_indices = None
        try:
            from causal_ssm_agent.models.likelihoods.graph_analysis import (
                kalman_block_profile_indices,
            )

            partition = runtime.inference_structure.first_pass_rb.partition
            if partition is not None and runtime.inference_structure.likelihood_path == "composed":
                kalman_indices = kalman_block_profile_indices(ssm_model.spec, partition)
                logger.info(
                    "First-pass RB plan: profiling %d/%d Kalman-block params (skipping particle block)",
                    len(kalman_indices),
                    ssm_model.spec.n_latent,
                )
        except Exception:
            logger.debug("Inference-structure profile filtering failed", exc_info=True)

        # Run profile likelihood check (only Kalman-block params when mixed)
        result = profile_likelihood(
            model=ssm_model,
            observations=observations,
            times=times,
            profile_indices=kalman_indices,
            n_grid=n_grid,
            confidence=confidence,
            sweep_context=sweep_context,
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
            "parametric_id": {
                "checked": True,
                "t_rule": t_rule.model_dump(),
                "sensitivity_analysis": sensitivity_payload,
                "summary": summary,
                "per_param_classification": per_param,
                "threshold": float(result.threshold),
                "n_parameters": len(result.parameter_names),
                "parameter_names": result.parameter_names,
            },
            "inference_structure": inference_structure_payload,
        }

    except Exception as e:
        logger.exception("Parametric ID check failed")
        return {
            "parametric_id": {
                "checked": False,
                "error": str(e),
            },
            "inference_structure": None,
        }


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
        stage4_result: Output from stage4_agentic_flow
        raw_data: Canonical observation rows (indicator, value, anchor_time, support metadata)
        builder: Pre-built SSMModelBuilder (avoids rebuilding)

    Returns:
        stage4_result augmented with 'parametric_id' and 'inference_structure' keys
    """
    model_spec = stage4_result["model_spec"]
    priors = stage4_result["priors"]
    causal_spec = stage4_result.get("causal_spec")
    compiled_ssm = stage4_result.get("_compiled_ssm")

    diagnostics = parametric_id_task(
        model_spec,
        priors,
        raw_data,
        _causal_spec=causal_spec,
        compiled_ssm=compiled_ssm,
        builder=builder,
    )

    return {
        **stage4_result,
        "parametric_id": diagnostics["parametric_id"],
        "inference_structure": diagnostics["inference_structure"],
    }
