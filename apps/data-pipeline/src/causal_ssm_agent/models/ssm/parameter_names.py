"""Shared semantic parameter-name resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from causal_ssm_agent.orchestrator.schemas_model import ModelSpec, ParameterRole

if TYPE_CHECKING:
    from collections.abc import Sequence

INITIAL_STATE_CORRELATION_PREFIXES = ("cor0_",)
INITIAL_STATE_CORRELATION_KEYWORDS = ["cor0"]
INITIAL_STATE_CORRELATION_PRIOR_DEFAULTS: dict[str, float] = {
    "mu": 0.0,
    "sigma": 0.5,
    "lower": -1.0,
    "upper": 1.0,
}


@dataclass(frozen=True)
class InitialStateCorrelationBinding:
    """Resolved authored initial-state correlation parameter."""

    parameter_name: str
    state1_name: str
    state2_name: str
    row: int
    col: int


def split_compound_name(
    compound: str,
    valid_first: set[str],
    valid_second: set[str],
) -> tuple[str, str] | None:
    """Split an underscore-joined name into two known names."""
    parts = compound.split("_")
    for idx in range(1, len(parts)):
        first = "_".join(parts[:idx])
        second = "_".join(parts[idx:])
        if first in valid_first and second in valid_second:
            return first, second
    return None


def _resolve_model_spec(model_spec: ModelSpec | dict) -> ModelSpec:
    return ModelSpec.model_validate(model_spec) if isinstance(model_spec, dict) else model_spec


def _strip_initial_state_correlation_prefix(name: str) -> str:
    for prefix in INITIAL_STATE_CORRELATION_PREFIXES:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def resolve_initial_state_correlation_bindings(
    latent_names: Sequence[str],
    model_spec: ModelSpec | dict,
) -> list[InitialStateCorrelationBinding]:
    """Resolve authored initial-state correlation parameters against latent names."""
    spec_obj = _resolve_model_spec(model_spec)
    latent_idx = {name: idx for idx, name in enumerate(latent_names)}
    latent_name_set = set(latent_idx)

    unresolved: list[tuple[str, str, str, int, int]] = []
    seen_pairs: dict[tuple[int, int], str] = {}

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.INITIAL_STATE_CORRELATION:
            continue
        compound = _strip_initial_state_correlation_prefix(parameter.name)
        result = split_compound_name(compound, latent_name_set, latent_name_set)
        if result is None:
            raise ValueError(
                "Could not parse INITIAL_STATE_CORRELATION parameter "
                f"{parameter.name!r} into known latent states {sorted(latent_name_set)}"
            )
        state1_name, state2_name = result
        idx1 = latent_idx[state1_name]
        idx2 = latent_idx[state2_name]
        if idx1 == idx2:
            raise ValueError(
                "INITIAL_STATE_CORRELATION parameters must reference two distinct latent "
                f"states; got self-correlation {parameter.name!r}"
            )
        row, col = (idx1, idx2) if idx1 > idx2 else (idx2, idx1)
        pair_key = (row, col)
        if pair_key in seen_pairs:
            raise ValueError(
                "Duplicate INITIAL_STATE_CORRELATION parameters target the same initial-state "
                f"pair ({latent_names[col]!r}, {latent_names[row]!r}): "
                f"{seen_pairs[pair_key]!r} and {parameter.name!r}"
            )
        seen_pairs[pair_key] = parameter.name
        unresolved.append((parameter.name, state1_name, state2_name, row, col))

    unresolved.sort(key=lambda entry: (entry[3], entry[4], entry[0]))
    return [
        InitialStateCorrelationBinding(
            parameter_name=parameter_name,
            state1_name=state1_name,
            state2_name=state2_name,
            row=row,
            col=col,
        )
        for parameter_name, state1_name, state2_name, row, col in unresolved
    ]


def build_initial_state_correlation_mask(
    latent_names: Sequence[str],
    model_spec: ModelSpec | dict,
) -> np.ndarray | None:
    """Build a sparse lower-triangle mask for authored initial-state correlations."""
    bindings = resolve_initial_state_correlation_bindings(latent_names, model_spec)
    if not bindings:
        return None

    mask = np.zeros((len(latent_names), len(latent_names)), dtype=bool)
    for binding in bindings:
        mask[binding.row, binding.col] = True
    return mask
