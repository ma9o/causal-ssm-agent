"""Partial drift guardrails for agentic Stage 4 prior authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nof1_causal_lab.artifacts.model_spec import ParameterRole
from nof1_causal_lab.flows import get_prefect_logger
from nof1_causal_lab.models.ssm.compile.artifact import validate_model_spec_for_compilation
from nof1_causal_lab.models.ssm.compile.inputs import translate_spec
from nof1_causal_lab.models.ssm.compile.prior_compilation import (
    PriorCompilationError,
    _positive_prior_mean_values,
    _prior_values_1d,
    _value_at,
    compile_priors,
    logm_diagnostic_mean_drift,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind
from nof1_causal_lab.workers.schemas_prior import (
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
    diagnostic_drift: np.ndarray | None = None

    @property
    def latent_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.latent_names)}

    def mean_drift(self) -> np.ndarray:
        if self.diagnostic_drift is not None:
            return self.diagnostic_drift.copy()

        drift = np.zeros((len(self.latent_names), len(self.latent_names)), dtype=float)
        for idx in np.flatnonzero(self.offdiag_present):
            effect_idx, cause_idx = self.offdiag_positions[idx]
            drift[effect_idx, cause_idx] = float(self.offdiag_mu[idx])
        for idx in np.flatnonzero(self.diag_present):
            drift[idx, idx] = -abs(float(self.diag_mu[idx]))
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
    drift_parameter_names = {
        parameter.name
        for parameter in resolved_model_spec.parameters
        if parameter.role in {ParameterRole.AR_COEFFICIENT, ParameterRole.FIXED_EFFECT}
    }
    drift_priors = {
        parameter_name: prior_spec
        for parameter_name, prior_spec in authored_priors.items()
        if parameter_name in drift_parameter_names
    }
    prior_registry, index_maps, _diagnostics = compile_priors(
        drift_priors,
        resolved_model_spec,
        ssm_spec,
        edge_lag_days=edge_lag_days,
        causal_spec=causal_spec,
    )
    diag_bindings = {
        parameter_name: binding
        for parameter_name, binding in index_maps.by_site_kind(SiteKind.DYNAMICS_DECAY).items()
        if parameter_name.startswith(("rho_", "ar_"))
    }
    offdiag_bindings = {}
    offdiag_positions: list[tuple[int, int]] = []
    for parameter_name, binding in index_maps.by_site_kind(SiteKind.DYNAMICS_WEIGHT).items():
        effect_idx = binding.effect_idx
        cause_idx = binding.cause_idx
        if parameter_name.startswith("beta_") and effect_idx is not None and cause_idx is not None:
            offdiag_bindings[parameter_name] = binding
            offdiag_positions.append((int(effect_idx), int(cause_idx)))

    latent_names = tuple(ssm_spec.latent_names or ())
    latent_index_by_name = {name: idx for idx, name in enumerate(latent_names)}
    diag_mu = np.zeros(len(latent_names), dtype=float)
    diag_sigma = np.zeros(len(latent_names), dtype=float)
    diag_present = np.zeros(len(latent_names), dtype=bool)
    diag_parameter_by_index: dict[int, str] = {}
    for parameter_name in drift_priors:
        binding = diag_bindings.get(parameter_name)
        if binding is None:
            continue
        if not binding.construct_names:
            continue
        latent_index = latent_index_by_name.get(binding.construct_names[0])
        if latent_index is None:
            continue
        decay_prior = prior_registry.priors_by_site[binding.site_name]
        decay_mu = _positive_prior_mean_values(decay_prior)
        diag_present[latent_index] = True
        diag_parameter_by_index[latent_index] = parameter_name
        diag_mu[latent_index] = _value_at(decay_mu, binding.flat_index, default=0.0)

    offdiag_mu = np.zeros(len(offdiag_positions), dtype=float)
    offdiag_sigma = np.zeros(len(offdiag_positions), dtype=float)
    offdiag_present = np.zeros(len(offdiag_positions), dtype=bool)
    offdiag_parameter_by_index: dict[int, str] = {}
    offdiag_index_by_parameter = {
        parameter_name: idx for idx, parameter_name in enumerate(offdiag_bindings)
    }
    for parameter_name in drift_priors:
        binding = offdiag_bindings.get(parameter_name)
        if binding is None:
            continue
        position_index = offdiag_index_by_parameter[parameter_name]
        weight_prior = prior_registry.priors_by_site[binding.site_name]
        weight_mu = _prior_values_1d(weight_prior.params.get("mu"))
        weight_sigma = _prior_values_1d(weight_prior.params.get("sigma"))
        offdiag_present[position_index] = True
        offdiag_parameter_by_index[position_index] = parameter_name
        offdiag_mu[position_index] = _value_at(weight_mu, binding.flat_index, default=0.0)
        offdiag_sigma[position_index] = _value_at(
            weight_sigma,
            binding.flat_index,
            default=0.0,
        )

    diagnostic_drift = logm_diagnostic_mean_drift(
        prior_registry,
        ssm_spec,
        edge_lag_days=edge_lag_days,
    )
    if diagnostic_drift is not None:
        for latent_index in np.flatnonzero(diag_present):
            diag_mu[latent_index] = abs(float(diagnostic_drift[latent_index, latent_index]))
        for position_index in np.flatnonzero(offdiag_present):
            effect_idx, cause_idx = offdiag_positions[position_index]
            offdiag_mu[position_index] = float(diagnostic_drift[effect_idx, cause_idx])

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
        diagnostic_drift=diagnostic_drift,
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
        code="partial_drift_stability",
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
