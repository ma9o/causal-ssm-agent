"""Partial drift guardrails for agentic Stage 4 prior authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.ssm.structure_runtime import SSMStructureRuntime
from causal_ssm_agent.models.ssm_compilation import translate_spec
from causal_ssm_agent.models.ssm_compiler import validate_model_spec_for_compilation
from causal_ssm_agent.models.ssm_prior_compilation import PriorCompilationError, compile_priors
from causal_ssm_agent.workers.schemas_prior import (
    PriorPathologyCertificate,
    PriorValidationResult,
)

logger = get_prefect_logger(__name__)
_DRIFT_EPSILON = 1e-6


@dataclass(frozen=True)
class EffectRowBudget:
    """Prompt-facing stability budget for one target construct row."""

    target_construct: str
    target_parameter: str | None
    diagonal_magnitude: float
    diagonal_lower_bound: float
    specified_incoming_edges: int
    total_incoming_edges: int
    used_abs_mean: float
    used_abs_upper: float
    remaining_abs_mean: float
    remaining_abs_upper: float


@dataclass(frozen=True)
class _PartialDriftState:
    latent_names: tuple[str, ...]
    diag_mu: np.ndarray
    diag_sigma: np.ndarray
    diag_present: np.ndarray
    diag_parameter_by_index: dict[int, str]
    offdiag_positions: tuple[tuple[int, int], ...]
    offdiag_mu: np.ndarray
    offdiag_sigma: np.ndarray
    offdiag_present: np.ndarray
    offdiag_parameter_by_index: dict[int, str]

    @property
    def latent_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.latent_names)}

    def mean_drift(self) -> np.ndarray:
        drift = np.zeros((len(self.latent_names), len(self.latent_names)), dtype=float)
        for idx in np.flatnonzero(self.diag_present):
            drift[idx, idx] = -abs(float(self.diag_mu[idx]))
        for idx in np.flatnonzero(self.offdiag_present):
            effect_idx, cause_idx = self.offdiag_positions[idx]
            drift[effect_idx, cause_idx] = float(self.offdiag_mu[idx])
        return drift

    def has_all_diagonals(self) -> bool:
        return bool(self.diag_present.all())


def _build_partial_drift_state(
    *,
    model_spec: dict[str, Any] | None,
    authored_priors: dict[str, dict[str, Any]] | None,
    causal_spec: dict[str, Any] | None,
) -> _PartialDriftState | None:
    if model_spec is None or not authored_priors:
        return None

    resolved_model_spec, errors = validate_model_spec_for_compilation(
        model_spec,
        causal_spec=causal_spec,
    )
    if resolved_model_spec is None:
        raise ValueError("ModelSpec failed compiler validation:\n" + "\n".join(errors))

    ssm_spec, edge_lag_days = translate_spec(resolved_model_spec, causal_spec)
    ssm_priors, index_maps, _diagnostics = compile_priors(
        authored_priors,
        resolved_model_spec,
        ssm_spec,
        edge_lag_days=edge_lag_days,
        causal_spec=causal_spec,
    )
    structure_runtime = SSMStructureRuntime(ssm_spec)
    (
        _offdiag_param_index,
        _lambda_param_index,
        diag_param_index,
        _diffusion_diag_param_index,
        _diffusion_offdiag_param_index,
        _t0_offdiag_param_index,
        _t0_mean_param_index,
        _t0_sd_param_index,
        _manifest_mean_param_index,
        _manifest_var_param_index,
        _cint_param_index,
        _static_state_sd_param_index,
        _observation_site_param_index,
    ) = index_maps

    latent_names = tuple(ssm_spec.latent_names or ())
    diag_mu = np.zeros(len(latent_names), dtype=float)
    diag_sigma = np.zeros(len(latent_names), dtype=float)
    diag_present = np.zeros(len(latent_names), dtype=bool)
    diag_parameter_by_index: dict[int, str] = {}
    for parameter_name in authored_priors:
        if parameter_name not in diag_param_index:
            continue
        _attr, flat_index = diag_param_index[parameter_name]
        latent_index = structure_runtime.drift_diag_positions[flat_index]
        diag_present[latent_index] = True
        diag_parameter_by_index[latent_index] = parameter_name
        diag_mu[latent_index] = float(ssm_priors.drift_diag.get("mu", [])[flat_index])
        diag_sigma[latent_index] = float(ssm_priors.drift_diag.get("sigma", [])[flat_index])

    offdiag_positions = list(structure_runtime.offdiag_positions)
    offdiag_mu = np.zeros(len(offdiag_positions), dtype=float)
    offdiag_sigma = np.zeros(len(offdiag_positions), dtype=float)
    offdiag_present = np.zeros(len(offdiag_positions), dtype=bool)
    offdiag_parameter_by_index: dict[int, str] = {}
    drift_offdiag_mu = np.asarray(ssm_priors.drift_offdiag.get("mu", []), dtype=float)
    drift_offdiag_sigma = np.asarray(ssm_priors.drift_offdiag.get("sigma", []), dtype=float)
    offdiag_param_index = index_maps[0]
    for parameter_name in authored_priors:
        if parameter_name not in offdiag_param_index:
            continue
        _attr, flat_index = offdiag_param_index[parameter_name]
        offdiag_present[flat_index] = True
        offdiag_parameter_by_index[flat_index] = parameter_name
        offdiag_mu[flat_index] = (
            float(drift_offdiag_mu[flat_index]) if flat_index < drift_offdiag_mu.size else 0.0
        )
        offdiag_sigma[flat_index] = (
            float(drift_offdiag_sigma[flat_index]) if flat_index < drift_offdiag_sigma.size else 0.0
        )

    return _PartialDriftState(
        latent_names=latent_names,
        diag_mu=diag_mu,
        diag_sigma=diag_sigma,
        diag_present=diag_present,
        diag_parameter_by_index=diag_parameter_by_index,
        offdiag_positions=tuple(offdiag_positions),
        offdiag_mu=offdiag_mu,
        offdiag_sigma=offdiag_sigma,
        offdiag_present=offdiag_present,
        offdiag_parameter_by_index=offdiag_parameter_by_index,
    )


def _incoming_effect_parameter_names(
    state: _PartialDriftState,
    target_construct: str,
) -> list[str]:
    """Return authored incoming effect parameters for one target row."""
    target_idx = state.latent_index.get(target_construct)
    if target_idx is None:
        return []

    parameter_names: list[str] = []
    for flat_index, (effect_idx, _cause_idx) in enumerate(state.offdiag_positions):
        if effect_idx != target_idx or not bool(state.offdiag_present[flat_index]):
            continue
        parameter_name = state.offdiag_parameter_by_index.get(flat_index)
        if parameter_name is not None:
            parameter_names.append(parameter_name)
    return parameter_names


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_row_budget_from_state(
    state: _PartialDriftState,
    target_construct: str,
) -> EffectRowBudget | None:
    """Compute the stability budget for one target row from a pre-built state."""
    target_idx = state.latent_index.get(target_construct)
    if target_idx is None or not bool(state.diag_present[target_idx]):
        return None

    row_indices = [
        flat_index
        for flat_index, (effect_idx, _cause_idx) in enumerate(state.offdiag_positions)
        if effect_idx == target_idx
    ]
    specified_indices = [
        flat_index for flat_index in row_indices if bool(state.offdiag_present[flat_index])
    ]
    diagonal_magnitude = abs(float(state.diag_mu[target_idx]))
    diagonal_lower_bound = diagonal_magnitude - float(state.diag_sigma[target_idx])
    used_abs_mean = float(
        sum(abs(float(state.offdiag_mu[flat_index])) for flat_index in specified_indices)
    )
    used_abs_upper = float(
        sum(
            abs(float(state.offdiag_mu[flat_index])) + float(state.offdiag_sigma[flat_index])
            for flat_index in specified_indices
        )
    )
    return EffectRowBudget(
        target_construct=target_construct,
        target_parameter=state.diag_parameter_by_index.get(target_idx),
        diagonal_magnitude=diagonal_magnitude,
        diagonal_lower_bound=diagonal_lower_bound,
        specified_incoming_edges=len(specified_indices),
        total_incoming_edges=len(row_indices),
        used_abs_mean=used_abs_mean,
        used_abs_upper=used_abs_upper,
        remaining_abs_mean=diagonal_magnitude - used_abs_mean,
        remaining_abs_upper=diagonal_lower_bound - used_abs_upper,
    )


def _check_eigenvalue_stability(
    state: _PartialDriftState,
    *,
    related_parameters: list[str],
    issue_context: str,
    suggested_adjustment: str,
    feedback_context: str,
) -> tuple[PriorValidationResult, str] | None:
    """Shared eigenvalue stability check for partial drift guards."""
    if not state.has_all_diagonals():
        return None

    max_real = float(np.max(np.real(np.linalg.eigvals(state.mean_drift()))))
    if max_real < 0:
        return None

    diagnostic = PriorValidationResult(
        parameter="dynamics_stability",
        is_valid=False,
        code="partial_dynamics_stability",
        origin="prior_predictive",
        issue=(
            "Partial drift guard detected an unstable mean CT operator "
            f"(max real eigenvalue={max_real:.4f}) {issue_context}."
        ),
        suggested_adjustment=suggested_adjustment,
        related_parameters=related_parameters,
        failure_stage="latent_dynamics",
        pathology_certificate=PriorPathologyCertificate(
            kind="dynamics_stability",
            primary_score=max_real,
        ),
    )
    feedback = "\n".join(
        [
            "PARTIAL DRIFT CHECK FAILED:",
            (
                f"- the compiled mean drift is already unstable {feedback_context} "
                f"(max real eigenvalue `{max_real:.4f}`)"
            ),
            "- repair this drift block now instead of deferring to the final prior-predictive gate",
        ]
    )
    return diagnostic, feedback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_effect_row_budget(
    *,
    model_spec: dict[str, Any] | None,
    authored_priors: dict[str, dict[str, Any]] | None,
    causal_spec: dict[str, Any] | None,
    target_construct: str,
) -> EffectRowBudget | None:
    """Return the compiled CT stability budget for one effect block target row."""
    try:
        state = _build_partial_drift_state(
            model_spec=model_spec,
            authored_priors=authored_priors,
            causal_spec=causal_spec,
        )
    except (PriorCompilationError, ValueError):
        logger.debug("Skipping effect-row budget rendering", exc_info=True)
        return None
    if state is None:
        return None

    return _build_row_budget_from_state(state, target_construct)


def validate_dynamics_block_partial_drift(
    *,
    model_spec: dict[str, Any] | None,
    authored_priors: dict[str, dict[str, Any]] | None,
    causal_spec: dict[str, Any] | None,
    active_construct_names: tuple[str, ...],
    active_parameter_names: tuple[str, ...],
) -> tuple[PriorValidationResult, str] | None:
    """Return an early drift diagnostic for one accepted dynamics block, if needed."""
    state = _build_partial_drift_state(
        model_spec=model_spec,
        authored_priors=authored_priors,
        causal_spec=causal_spec,
    )
    if state is None:
        return None

    budgets = [
        budget
        for construct_name in active_construct_names
        if (budget := _build_row_budget_from_state(state, construct_name)) is not None
    ]
    if not budgets:
        return None

    weakest_budget = min(budgets, key=lambda budget: budget.diagonal_lower_bound)
    related_parameters = list(
        dict.fromkeys(
            [weakest_budget.target_parameter, *active_parameter_names]
            if weakest_budget.target_parameter
            else active_parameter_names
        )
    )

    coupled_effect_parameters = [
        parameter_name
        for construct_name in active_construct_names
        for parameter_name in _incoming_effect_parameter_names(state, construct_name)
    ]
    return _check_eigenvalue_stability(
        state,
        related_parameters=list(dict.fromkeys([*related_parameters, *coupled_effect_parameters])),
        issue_context=(
            f"after updating the dynamics block for {', '.join(active_construct_names)}"
        ),
        suggested_adjustment=(
            "Tighten the active dynamics priors toward faster decay before keeping the current "
            "drift subsystem. Use the reported row headroom as guidance, but treat actual drift "
            "instability as the hard stop."
        ),
        feedback_context="after this dynamics update",
    )


def validate_effect_block_partial_drift(
    *,
    model_spec: dict[str, Any] | None,
    authored_priors: dict[str, dict[str, Any]] | None,
    causal_spec: dict[str, Any] | None,
    target_construct: str,
    active_parameter_names: tuple[str, ...],
) -> tuple[PriorValidationResult, str] | None:
    """Return an early drift diagnostic for one accepted effect block, if needed."""
    state = _build_partial_drift_state(
        model_spec=model_spec,
        authored_priors=authored_priors,
        causal_spec=causal_spec,
    )
    if state is None:
        return None

    budget = _build_row_budget_from_state(state, target_construct)
    if budget is None:
        return None

    related_parameters = list(
        dict.fromkeys(
            [budget.target_parameter, *active_parameter_names]
            if budget.target_parameter
            else active_parameter_names
        )
    )

    return _check_eigenvalue_stability(
        state,
        related_parameters=related_parameters,
        issue_context=f"after updating {target_construct}",
        suggested_adjustment=(
            f"Shrink the incoming beta priors on {target_construct} or tighten "
            f"{budget.target_parameter or f'rho_{target_construct}'} toward faster decay. "
            "Use the reported row headroom as guidance, but treat actual drift instability as "
            "the hard stop."
        ),
        feedback_context=f"after `{target_construct}`",
    )
