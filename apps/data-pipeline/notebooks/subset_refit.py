"""Utilities for refitting a Stage 5b model on a retained construct subset."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import polars as pl

from causal_ssm_agent.flows.stages.stage4.agentic.stage4_parameter_surfaces import (
    parameter_is_active_for_model_spec,
)
from causal_ssm_agent.flows.stages.stage4.agentic.stage4_skeleton import (
    derive_deterministic_spec,
)
from causal_ssm_agent.models.ssm_builder import PreparedModelRuntime, prepare_model_runtime
from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact
from causal_ssm_agent.utils.causal_spec import (
    get_constructs,
    get_edges,
    get_estimation_edges,
    get_estimation_state_order,
    get_indicators,
    get_induced_dependencies,
)


@dataclass(frozen=True)
class Stage5bSubsetRuntime:
    """Compiled and prepared model inputs for a Stage 5b subset refit."""

    causal_spec: dict[str, Any]
    model_spec: dict[str, Any]
    authored_priors: dict[str, dict[str, Any]]
    compiled_ssm: dict[str, Any]
    data_for_model: pl.DataFrame
    runtime: PreparedModelRuntime
    retained_constructs: tuple[str, ...]
    dropped_constructs: tuple[str, ...]
    retained_indicators: tuple[str, ...]
    retained_parameters: tuple[str, ...]


def _public_parameter_row(parameter: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(parameter["name"]),
        "role": str(parameter["role"]),
        "constraint": str(parameter["constraint"]),
        "description": str(parameter["description"]),
    }


def _validate_requested_constructs(
    requested: set[str],
    *,
    known_constructs: set[str],
    label: str,
) -> None:
    unknown = sorted(requested - known_constructs)
    if unknown:
        raise ValueError(
            f"{label} references constructs absent from estimation.state_order: {unknown}"
        )


def resolve_retained_constructs(
    causal_spec: dict[str, Any],
    *,
    keep_constructs: set[str] | list[str] | tuple[str, ...] | None = None,
    drop_constructs: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve ordered retained and dropped estimation constructs."""
    state_order = tuple(get_estimation_state_order(causal_spec))
    known_constructs = set(state_order)
    if not state_order:
        raise ValueError("causal_spec.estimation.state_order is empty")

    keep_set = set(keep_constructs) if keep_constructs is not None else known_constructs
    drop_set = set(drop_constructs or ())
    _validate_requested_constructs(keep_set, known_constructs=known_constructs, label="keep")
    _validate_requested_constructs(drop_set, known_constructs=known_constructs, label="drop")

    retained = tuple(name for name in state_order if name in keep_set and name not in drop_set)
    dropped = tuple(name for name in state_order if name not in retained)
    if not retained:
        raise ValueError("Stage 5b subset retains no constructs")
    return retained, dropped


def _subset_identifiability(
    identifiability: dict[str, Any],
    retained_constructs: set[str],
) -> dict[str, Any]:
    subset = deepcopy(identifiability)
    for key in ("identifiable_treatments", "non_identifiable_treatments"):
        value = subset.get(key)
        if isinstance(value, dict):
            subset[key] = {
                name: payload for name, payload in value.items() if name in retained_constructs
            }
    for key in ("estimable_treatments", "all_treatments"):
        value = subset.get(key)
        if isinstance(value, list):
            subset[key] = [name for name in value if name in retained_constructs]
    return subset


def subset_causal_spec(
    causal_spec: dict[str, Any],
    *,
    keep_constructs: set[str] | list[str] | tuple[str, ...] | None = None,
    drop_constructs: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Drop unretained constructs plus all incident latent and estimation edges."""
    retained, dropped = resolve_retained_constructs(
        causal_spec,
        keep_constructs=keep_constructs,
        drop_constructs=drop_constructs,
    )
    retained_set = set(retained)

    subset = deepcopy(causal_spec)
    subset["latent"]["constructs"] = [
        construct for construct in get_constructs(causal_spec) if construct["name"] in retained_set
    ]
    subset["latent"]["edges"] = [
        edge
        for edge in get_edges(causal_spec)
        if edge["cause"] in retained_set and edge["effect"] in retained_set
    ]
    subset["measurement"]["indicators"] = [
        indicator
        for indicator in get_indicators(causal_spec)
        if indicator["construct_name"] in retained_set
    ]
    subset["estimation"]["state_order"] = list(retained)
    subset["estimation"]["edges"] = [
        edge
        for edge in get_estimation_edges(causal_spec)
        if edge["cause"] in retained_set and edge["effect"] in retained_set
    ]
    subset["estimation"]["induced_dependencies"] = [
        dependency
        for dependency in get_induced_dependencies(causal_spec)
        if set(dependency["between"]).issubset(retained_set)
        and set(dependency["source_confounders"]).issubset(retained_set)
    ]
    if isinstance(subset.get("identifiability"), dict):
        subset["identifiability"] = _subset_identifiability(
            subset["identifiability"],
            retained_set,
        )
    return subset, retained, dropped


def subset_model_spec(
    model_spec: dict[str, Any],
    causal_spec: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the active ModelSpec surface for a filtered causal spec."""
    retained_indicator_names = {indicator["name"] for indicator in get_indicators(causal_spec)}
    likelihoods = [
        deepcopy(likelihood)
        for likelihood in model_spec["likelihoods"]
        if likelihood["variable"] in retained_indicator_names
    ]
    chosen_likelihood_by_variable = {
        likelihood["variable"]: likelihood for likelihood in likelihoods
    }

    initialization_policy = model_spec["initialization_policy"]
    observation_intercept_policy = model_spec["observation_intercept_policy"]
    equilibrium_forcing = bool(model_spec["equilibrium_forcing"])
    original_parameter_by_name = {
        parameter["name"]: parameter for parameter in model_spec["parameters"]
    }

    active_parameter_names: list[str] = []
    for parameter in derive_deterministic_spec(causal_spec).all_params:
        if not parameter_is_active_for_model_spec(
            parameter,
            chosen_likelihood_by_variable,
            initialization_policy=initialization_policy,
            observation_intercept_policy=observation_intercept_policy,
            equilibrium_forcing=equilibrium_forcing,
        ):
            continue
        active_parameter_names.append(parameter["name"])

    missing_parameters = sorted(set(active_parameter_names) - set(original_parameter_by_name))
    if missing_parameters:
        raise ValueError(
            "Subset ModelSpec requires original Stage 4 parameter rows for active parameters: "
            f"{missing_parameters}"
        )

    parameters = [
        _public_parameter_row(original_parameter_by_name[name]) for name in active_parameter_names
    ]

    return {
        "likelihoods": likelihoods,
        "parameters": parameters,
        "initialization_policy": initialization_policy,
        "observation_intercept_policy": observation_intercept_policy,
        "equilibrium_forcing": equilibrium_forcing,
    }


def subset_authored_priors(
    authored_priors: dict[str, dict[str, Any]],
    parameter_names: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Keep authored priors for the retained active parameter set."""
    missing_priors = [name for name in parameter_names if name not in authored_priors]
    if missing_priors:
        raise ValueError(
            "Subset Stage 5b refit requires authored priors for active parameters: "
            f"{missing_priors}"
        )
    return {name: deepcopy(authored_priors[name]) for name in parameter_names}


def subset_data_for_model(
    data_for_model: pl.DataFrame, causal_spec: dict[str, Any]
) -> pl.DataFrame:
    """Keep observation rows whose indicators remain in the filtered measurement model."""
    retained_indicators = [indicator["name"] for indicator in get_indicators(causal_spec)]
    return data_for_model.filter(pl.col("indicator").is_in(retained_indicators))


def build_stage5b_subset_runtime(
    *,
    causal_spec: dict[str, Any],
    model_spec: dict[str, Any],
    authored_priors: dict[str, dict[str, Any]],
    data_for_model: pl.DataFrame,
    keep_constructs: set[str] | list[str] | tuple[str, ...] | None = None,
    drop_constructs: set[str] | list[str] | tuple[str, ...] | None = None,
    sampler_config: dict[str, Any] | None = None,
) -> Stage5bSubsetRuntime:
    """Subset, recompile, and prepare the standard Stage 5b runtime."""
    subset_spec, retained_constructs, dropped_constructs = subset_causal_spec(
        causal_spec,
        keep_constructs=keep_constructs,
        drop_constructs=drop_constructs,
    )
    subset_model = subset_model_spec(model_spec, subset_spec)
    retained_parameters = tuple(parameter["name"] for parameter in subset_model["parameters"])
    subset_priors = subset_authored_priors(authored_priors, retained_parameters)
    subset_data = subset_data_for_model(data_for_model, subset_spec)
    compiled_ssm = compile_ssm_artifact(
        subset_model,
        subset_priors,
        causal_spec=subset_spec,
    )
    runtime = prepare_model_runtime(
        data_for_model=subset_data,
        compiled_ssm=compiled_ssm,
        sampler_config=sampler_config,
    )
    retained_indicators = tuple(indicator["name"] for indicator in get_indicators(subset_spec))
    return Stage5bSubsetRuntime(
        causal_spec=subset_spec,
        model_spec=subset_model,
        authored_priors=subset_priors,
        compiled_ssm=compiled_ssm,
        data_for_model=subset_data,
        runtime=runtime,
        retained_constructs=retained_constructs,
        dropped_constructs=dropped_constructs,
        retained_indicators=retained_indicators,
        retained_parameters=retained_parameters,
    )
