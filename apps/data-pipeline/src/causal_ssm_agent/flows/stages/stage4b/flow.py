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

from ... import get_prefect_logger
from ...run_store import load_parquet
from ...runtime_events import emit_nested_stage_running_event

logger = get_prefect_logger(__name__)

_SUBSTANTIVE_PROFILE_PREFIXES = (
    "cint_pop[",
    "drift_diag_pop[",
    "drift_offdiag_pop[",
    "lambda_free",
)


def _build_parametric_id_summary(
    profile_summary: dict[str, str] | None,
    sensitivity_payload: dict[str, Any] | None,
) -> dict[str, list[str]]:
    """Aggregate detailed diagnostics into the public Stage 4b summary shape."""
    structural_issues = sorted(
        name
        for name, classification in (profile_summary or {}).items()
        if classification == "structurally_unidentifiable"
    )
    weak_params = sorted(
        name
        for name, classification in (profile_summary or {}).items()
        if classification == "practically_unidentifiable"
    )

    # Sensitivity analysis exposes additional local identifiability warnings
    # even when profile likelihood does not classify a parameter as structural.
    for entry in (sensitivity_payload or {}).get("per_parameter", []):
        name = entry.get("parameter")
        if not name or name in structural_issues or name in weak_params:
            continue
        if entry.get("sv_status") in {"warn", "fail"} or entry.get("normalized_sv_status") in {
            "warn",
            "fail",
        }:
            weak_params.append(name)

    return {
        "structural_issues": structural_issues,
        "boundary_issues": [],
        "weak_params": sorted(weak_params),
    }


def _is_substantive_profile_parameter(name: str) -> bool:
    """Return whether a scalar parameter belongs to the substantive drift/loading core."""
    return any(name.startswith(prefix) for prefix in _SUBSTANTIVE_PROFILE_PREFIXES)


def _select_profile_indices_from_sensitivity(
    sensitivity_payload: dict[str, Any] | None,
    *,
    scalar_names: list[str] | None,
    default_indices: list[int] | None,
) -> list[int] | None:
    """Choose which scalar parameters should escalate from sensitivity to profiling.

    Profile likelihood is reserved for substantive raw-sensitivity failures.
    Scale/covariance nuisance terms often trip the normalized sensitivity
    heuristic without justifying an expensive global profile sweep.
    """
    if sensitivity_payload is None or scalar_names is None:
        return default_indices

    substantive_failures = {
        entry["parameter"]
        for entry in sensitivity_payload.get("per_parameter", [])
        if entry.get("sv_status") == "fail"
        and _is_substantive_profile_parameter(entry.get("parameter", ""))
    }
    if not substantive_failures:
        return []

    selected = [idx for idx, name in enumerate(scalar_names) if name in substantive_failures]
    if default_indices is not None:
        allowed = set(default_indices)
        selected = [idx for idx in selected if idx in allowed]
    return selected


@task(task_run_name="parametric-id-check")
def parametric_id_task(
    data_for_model: pl.DataFrame,
    n_grid: int = 20,
    confidence: float = 0.95,
    compiled_ssm: dict | None = None,
    builder: Any = None,
) -> dict:
    """Run parametric identifiability checks via profile likelihood.

    1. Build SSMModel from compiled artifact (or reuse provided builder)
    2. Prepare data (pivot raw -> wide)
    3. Call profile_likelihood()
    4. Return result summary

    Args:
        data_for_model: Canonical observation rows (indicator, value, anchor_time, support metadata)
        n_grid: Number of grid points for profile likelihood
        confidence: Confidence level for chi-squared threshold
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
            data_for_model=data_for_model,
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

        # T-rule: fast conservative screen surfaced as a warning if it fails
        from causal_ssm_agent.utils.parametric_id import check_t_rule

        t_rule = check_t_rule(ssm_model.spec, T=T)
        t_rule.print_report()

        if not t_rule.satisfies:
            return {
                "parametric_id": {
                    "checked": True,
                    "t_rule": t_rule.model_dump(),
                    "summary": {
                        "structural_issues": [],
                        "boundary_issues": [],
                        "weak_params": [],
                    },
                    "error": (
                        f"T-rule warning: {t_rule.n_free_params} free params "
                        f"> conservative lower-bound {t_rule.n_moments} moment conditions. "
                        "This screen is warning-only and does not halt inference."
                    ),
                },
                "inference_structure": inference_structure_payload,
            }

        from causal_ssm_agent.utils.parametric_id import get_stage4b_sweep_context

        sweep_context = get_stage4b_sweep_context(ssm_model)

        # Sensitivity analysis: structural check (sufficient for local identifiability)
        sensitivity_payload = None
        try:
            from causal_ssm_agent.utils.parametric_id import (
                OutputSensitivityUnsupportedError,
                output_sensitivity_analysis,
            )

            sa_result = output_sensitivity_analysis(
                ssm_model,
                times,
                observations=observations,
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
        except OutputSensitivityUnsupportedError as exc:
            logger.info("Sensitivity analysis unavailable for this Stage 4b model: %s", exc)
        except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as exc:
            logger.warning(
                "Sensitivity analysis failed, continuing with profile likelihood: %s", exc
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
        except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as exc:
            logger.warning("Inference-structure profile filtering failed: %s", exc)

        if runtime.inference_structure.likelihood_path == "particle":
            logger.info(
                "Stage 4b: skipping profile likelihood on particle-only path; "
                "sensitivity analysis is the terminal diagnostic"
            )
            profile_indices = []
        else:
            profile_indices = _select_profile_indices_from_sensitivity(
                sensitivity_payload,
                scalar_names=getattr(sweep_context, "scalar_names", None),
                default_indices=kalman_indices,
            )

        profile_summary = None
        per_param = None
        threshold = None
        if profile_indices == []:
            logger.info(
                "Stage 4b: skipping profile likelihood because sensitivity found no substantive raw failures"
            )
        else:
            if profile_indices is not None:
                logger.info(
                    "Stage 4b: profiling %d substantive parameter(s) after sensitivity gating",
                    len(profile_indices),
                )
            result = profile_likelihood(
                model=ssm_model,
                observations=observations,
                times=times,
                profile_indices=profile_indices,
                n_grid=n_grid,
                confidence=confidence,
                sweep_context=sweep_context,
            )

            result.print_report()
            profile_summary = result.summary()
            threshold = float(result.threshold)

            # Build per-parameter classifications with profile curve data
            per_param = []
            for name in result.parameter_names:
                profile = result.parameter_profiles[name]
                classification = profile_summary[name]
                peak_ll = float(jnp.max(profile["profile_ll"]))
                per_param.append(
                    {
                        "name": name,
                        "classification": classification,
                        "profile_x": [float(v) for v in profile["grid_con"]],
                        "profile_ll": [float(v) - peak_ll for v in profile["profile_ll"]],
                    }
                )

        summary = _build_parametric_id_summary(profile_summary, sensitivity_payload)

        return {
            "parametric_id": {
                "checked": True,
                "t_rule": t_rule.model_dump(),
                "sensitivity_analysis": sensitivity_payload,
                "summary": summary,
                "per_param_classification": per_param,
                "threshold": threshold,
            },
            "inference_structure": inference_structure_payload,
        }

    except (ValueError, RuntimeError, FloatingPointError, ArithmeticError) as e:
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
    compiled_ssm: dict | None,
    data_for_model: pl.DataFrame,
    builder: Any = None,
    root_run_id: str | None = None,
) -> dict:
    """Stage 4b: Parametric identifiability check.

    Runs pre-fit diagnostics on the compiled SSM artifact.

    Args:
        compiled_ssm: Serialized executable artifact from stage 4
        data_for_model: Canonical observation rows (indicator, value, anchor_time, support metadata)
        builder: Pre-built SSMModelBuilder (avoids rebuilding)

    Returns:
        Dict with 'parametric_id' and 'inference_structure' keys
    """
    if root_run_id:
        emit_nested_stage_running_event(root_run_id, "stage-4b")

    diagnostics = parametric_id_task(
        data_for_model,
        compiled_ssm=compiled_ssm,
        builder=builder,
    )

    return {
        "parametric_id": diagnostics["parametric_id"],
        "inference_structure": diagnostics["inference_structure"],
    }


def run_stage4b(
    stage4: dict,
    stage2: dict,
    ssm_builder: Any = None,
    root_run_id: str | None = None,
) -> dict:
    """Run Stage 4b and materialize its public outcome."""
    result = stage4b_parametric_id_flow(
        compiled_ssm=stage4.get("_compiled_ssm"),
        data_for_model=load_parquet(stage2["_data_for_model_path"]),
        builder=ssm_builder,
        root_run_id=root_run_id,
    )
    param_id = result.get("parametric_id") or {}
    t_rule: dict[str, Any] = {}

    if param_id.get("checked", False):
        t_rule = param_id.get("t_rule", {})
        if not t_rule.get("satisfies", True):
            logger.warning(
                "Stage 4b warning: T-rule screen failed (%s free params > conservative lower-bound %s moments), continuing",
                t_rule.get("n_free_params"),
                t_rule.get("n_moments"),
            )
        summary = param_id.get("summary", {})
        if summary.get("structural_issues"):
            logger.warning(
                "STRUCTURAL non-identifiability detected — some parameters unconstrained"
            )
        elif summary.get("boundary_issues"):
            logger.warning("Boundary identifiability issues at some prior draws")
        else:
            logger.info("Parametric identifiability OK")
        weak = summary.get("weak_params", [])
        if weak:
            logger.info("  Weak parameters (low contraction): %s", weak)
    else:
        logger.info("  Skipped: %s", param_id.get("error", "unknown"))

    if param_id.get("checked", False):
        summary = param_id.get("summary", {})
        has_issues = (
            not t_rule.get("satisfies", True)
            or summary.get("structural_issues")
            or summary.get("boundary_issues")
            or summary.get("weak_params")
        )
        outcome = "warn" if has_issues else "success"
    else:
        outcome = "success"

    result["outcome"] = outcome
    return result
