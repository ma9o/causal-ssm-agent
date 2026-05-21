"""Stage 4 deterministic skeleton: parameter enumeration and likelihood resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nof1_causal_lab.artifacts.model_spec import VALID_LINKS_FOR_DISTRIBUTION
from nof1_causal_lab.distributions import VALID_LIKELIHOODS_FOR_DTYPE, DistributionFamily
from nof1_causal_lab.utils.causal_spec import (
    build_reference_indicator_lookup,
    get_constructs,
    get_estimation_edges,
    get_estimation_state_order,
    get_indicator_polarity,
    get_induced_dependencies,
    get_manifest_indicators,
    get_marginalized_scales,
)
from nof1_causal_lab.utils.observation_semantics import get_observation_semantics

from .stage4_parameter_surfaces import parameter_is_active_for_model_spec


@dataclass(frozen=True)
class Stage4Skeleton:
    """Deterministic Stage 4 decision surface derived from the causal spec."""

    resolved_likelihoods: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_indicators: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    loading_params: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_params(self) -> list[dict[str, Any]]:
        """Return the full final parameter inventory, including loadings."""
        return [*self.parameters, *self.loading_params]

    @property
    def final_parameter_names(self) -> list[str]:
        """Return the final parameter names in deterministic order."""
        return [param["name"] for param in self.all_params]


def derive_deterministic_spec(causal_spec: dict) -> Stage4Skeleton:
    """Pre-compute all deterministic parts of the stage-4 model skeleton."""
    retained_state_order = get_estimation_state_order(causal_spec)
    retained_edges = get_estimation_edges(causal_spec)
    induced_dependencies = get_induced_dependencies(causal_spec)
    indicators = get_manifest_indicators(causal_spec)
    latent_construct_lookup = {
        construct["name"]: construct for construct in get_constructs(causal_spec)
    }
    retained_constructs = [
        latent_construct_lookup[name]
        for name in retained_state_order
        if name in latent_construct_lookup
    ]

    grouped_indicators = indicators_per_construct(indicators)
    reference_indicator_lookup = build_reference_indicator_lookup(indicators)
    retained_construct_names = {construct["name"] for construct in retained_constructs}

    resolved_likelihoods: list[dict[str, Any]] = []
    ambiguous_indicators: list[dict[str, Any]] = []
    seed_parameters: list[dict[str, Any]] = []
    seed_loading_params: list[dict[str, Any]] = []

    # --- Likelihoods ---
    for indicator in indicators:
        name = indicator["name"]
        dtype = indicator.get("measurement_dtype", "continuous")
        valid_dists = VALID_LIKELIHOODS_FOR_DTYPE.get(dtype, ())
        indicator_semantics = _indicator_semantics_fields(indicator)

        if len(valid_dists) == 1:
            dist = next(iter(valid_dists))
            valid_links = VALID_LINKS_FOR_DISTRIBUTION[dist]
            if len(valid_links) == 1:
                link = next(iter(valid_links))
                resolved_likelihoods.append(
                    {
                        "variable": name,
                        "construct_name": indicator.get("construct_name"),
                        "distribution": dist.value,
                        "link": link.value,
                        **indicator_semantics,
                        "reasoning": f"{dtype} dtype -> {dist.value} / {link.value}",
                    }
                )
            else:
                ambiguous_indicators.append(
                    {
                        "variable": name,
                        "construct_name": indicator.get("construct_name"),
                        "dtype": dtype,
                        **indicator_semantics,
                        "fixed_distribution": dist.value,
                        "valid_links": sorted(link_fn.value for link_fn in valid_links),
                    }
                )
        else:
            link_options: dict[str, list[str]] = {}
            for distribution in sorted(valid_dists, key=lambda item: item.value):
                links = VALID_LINKS_FOR_DISTRIBUTION[distribution]
                link_options[distribution.value] = sorted(link_fn.value for link_fn in links)
            ambiguous_indicators.append(
                {
                    "variable": name,
                    "construct_name": indicator.get("construct_name"),
                    "dtype": dtype,
                    **indicator_semantics,
                    "valid_distributions": sorted(dist.value for dist in valid_dists),
                    "link_options": link_options,
                }
            )

    # --- Autoregressive parameters ---
    for construct in retained_constructs:
        if construct.get("temporal_status") == "time_varying":
            construct_name = construct["name"]
            seed_parameters.append(
                {
                    "name": f"rho_{construct_name}",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": (
                        "Baseline discrete-time persistence absent incoming feedback for "
                        f"{construct_name}"
                    ),
                    "construct": construct_name,
                }
            )

    # --- Fixed effects ---
    for edge in retained_edges:
        cause = edge["cause"]
        effect = edge["effect"]
        effect_construct = latent_construct_lookup.get(effect)
        if (
            effect_construct is not None
            and effect_construct.get("temporal_status") == "time_invariant"
        ):
            continue
        seed_parameters.append(
            {
                "name": f"beta_{cause}_{effect}",
                "role": "fixed_effect",
                "constraint": "none",
                "description": f"Effect of {cause} on {effect}",
                "cause": cause,
                "effect": effect,
                "lagged": edge.get("lagged", True),
            }
        )

    # --- Residual SDs ---
    for construct in retained_constructs:
        if construct.get("temporal_status") == "time_invariant":
            continue
        construct_name = construct["name"]
        seed_parameters.append(
            {
                "name": f"sigma_{construct_name}",
                "role": "residual_sd",
                "constraint": "positive",
                "description": f"Residual/innovation SD for {construct_name}",
                "construct": construct_name,
            }
        )

    seed_parameters.extend(_candidate_state_intercept_parameters(retained_constructs))
    seed_parameters.extend(_candidate_initial_state_parameters(retained_constructs))
    seed_parameters.extend(
        _measurement_error_parameters(
            indicators,
            retained_construct_names=retained_construct_names,
            indicators_per_construct=grouped_indicators,
        )
    )

    # --- Loadings ---
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        if (
            not construct_name
            or construct_name not in grouped_indicators
            or len(grouped_indicators[construct_name]) <= 1
        ):
            continue

        if indicator["name"] == reference_indicator_lookup.get(construct_name):
            continue

        reference_indicator = reference_indicator_lookup.get(construct_name)
        seed_loading_params.append(
            {
                "name": f"lambda_{indicator['name']}_{construct_name}",
                "role": "loading",
                "constraint": get_indicator_polarity(indicator),
                "description": f"Factor loading for {indicator['name']} on {construct_name}",
                "indicator": indicator["name"],
                "construct": construct_name,
                "reference_indicator": reference_indicator,
                "indicator_polarity": get_indicator_polarity(indicator),
            }
        )

    seed_parameters.extend(_candidate_observation_intercept_parameters(indicators))

    seed_parameters.extend(
        _candidate_observation_extra_parameters(
            indicators,
            resolved_likelihoods=resolved_likelihoods,
            ambiguous_indicators=ambiguous_indicators,
        )
    )

    seed_parameters.extend(
        _confounder_baseline_factor_parameters(
            get_marginalized_scales(causal_spec),
            retained_state_order=retained_state_order,
        )
    )

    # --- Correlations from marginalized confounders ---
    for dependency in induced_dependencies:
        if dependency["kind"] != "innovation_correlation":
            continue
        construct_1, construct_2 = dependency["between"]
        dependency_kind = dependency["kind"]
        seed_parameters.append(
            {
                "name": f"cor_{construct_1}_{construct_2}",
                "role": "correlation",
                "constraint": "correlation",
                "description": (
                    f"{dependency_kind.replace('_', ' ')} between {construct_1} and {construct_2} "
                    f"(source confounders: {', '.join(dependency['source_confounders'])})"
                ),
                "construct_1": construct_1,
                "construct_2": construct_2,
                "dependency_kind": dependency_kind,
                "source_confounders": dependency["source_confounders"],
            }
        )

    parameters, loading_params = _compiler_authoritative_stage4_inventory(
        causal_spec,
        resolved_likelihoods=resolved_likelihoods,
        ambiguous_indicators=ambiguous_indicators,
        seed_parameters=seed_parameters,
        seed_loading_params=seed_loading_params,
        retained_state_order=retained_state_order,
        retained_edges=retained_edges,
        induced_dependencies=induced_dependencies,
        retained_construct_names=retained_construct_names,
    )

    return Stage4Skeleton(
        resolved_likelihoods=resolved_likelihoods,
        ambiguous_indicators=ambiguous_indicators,
        parameters=parameters,
        loading_params=loading_params,
    )


def indicators_per_construct(indicators: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        if construct_name:
            grouped.setdefault(construct_name, []).append(indicator["name"])
    return grouped


def _indicator_semantics_fields(indicator: dict[str, Any]) -> dict[str, str | None]:
    """Return canonical support semantics for an indicator dict."""
    support_kind = indicator.get("support_kind")
    summary_operator = indicator.get("summary_operator")
    if isinstance(support_kind, str) and isinstance(summary_operator, str):
        return {
            "support_kind": support_kind,
            "summary_operator": summary_operator,
        }

    semantics = get_observation_semantics(indicator)
    return {
        "support_kind": semantics.support_kind.value,
        "summary_operator": semantics.summary_operator.value,
    }


def _compiler_authoritative_stage4_inventory(
    causal_spec: dict,
    *,
    resolved_likelihoods: list[dict[str, Any]],
    ambiguous_indicators: list[dict[str, Any]],
    seed_parameters: list[dict[str, Any]],
    seed_loading_params: list[dict[str, Any]],
    retained_state_order: list[str],
    retained_edges: list[dict[str, Any]],
    induced_dependencies: list[dict[str, Any]],
    retained_construct_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the compiler-authoritative public Stage 4 prior inventory."""
    from nof1_causal_lab.models.ssm.compile.artifact import (
        compile_ssm_artifact,
        resolve_prior_proposals,
    )

    seed_by_name = {
        parameter["name"]: dict(parameter) for parameter in [*seed_parameters, *seed_loading_params]
    }
    provisional_likelihoods = [
        *resolved_likelihoods,
        *_provisional_likelihood_choices(ambiguous_indicators),
    ]
    provisional_likelihood_by_variable = {
        str(likelihood["variable"]): dict(likelihood) for likelihood in provisional_likelihoods
    }
    provisional_model_spec = {
        "likelihoods": provisional_likelihoods,
        "initialization_policy": "stationary",
        "observation_intercept_policy": "free",
        "equilibrium_forcing": False,
        "parameters": [
            parameter
            for parameter in [*seed_parameters, *seed_loading_params]
            if parameter_is_active_for_model_spec(
                parameter,
                provisional_likelihood_by_variable,
                initialization_policy="stationary",
                observation_intercept_policy="free",
                equilibrium_forcing=False,
            )
        ],
    }
    try:
        compiled_ssm = compile_ssm_artifact(provisional_model_spec, {}, causal_spec=causal_spec)
    except ValueError:
        # Some unit tests intentionally exercise pre-compile-invalid causal specs.
        # Preserve the deterministic skeleton for those cases and simply skip the
        # compiler-backed membership step rather than failing at prompt-construction time.
        fallback_inventory = dict(seed_by_name)
        for parameter in _fallback_initial_state_parameters(retained_state_order):
            fallback_inventory.setdefault(parameter["name"], parameter)
        return _order_stage4_inventory(
            fallback_inventory.values(),
            retained_state_order=retained_state_order,
            retained_edges=retained_edges,
            induced_dependencies=induced_dependencies,
        )

    binding_by_parameter = {
        str(binding.get("parameter") or ""): dict(binding)
        for binding in list(compiled_ssm.get("parameter_bindings", []) or [])
        if isinstance(binding, dict) and binding.get("parameter")
    }

    final_inventory: dict[str, dict[str, Any]] = {}
    for row in resolve_prior_proposals(compiled_ssm, authored_priors={}):
        parameter_name = str(row.get("parameter") or "")
        if not parameter_name or _is_compiler_default_only_parameter_name(parameter_name):
            continue
        parameter = seed_by_name.get(parameter_name)
        if parameter is None:
            parameter = _parameter_metadata_from_compiler_row(
                parameter_name,
                binding=binding_by_parameter.get(parameter_name),
                retained_construct_names=retained_construct_names,
            )
        if parameter is None:
            raise ValueError(
                "Stage 4 deterministic inventory is missing compiler-exposed parameter "
                f"{parameter_name!r}; add explicit metadata instead of silently dropping it."
            )
        final_inventory[parameter_name] = _enrich_parameter_with_binding(
            parameter,
            binding_by_parameter.get(parameter_name),
        )

    for parameter_name, parameter in seed_by_name.items():
        if parameter_name in final_inventory or not _is_conditional_prior_surface_parameter(
            parameter
        ):
            continue
        final_inventory[parameter_name] = _enrich_parameter_with_binding(
            parameter,
            binding_by_parameter.get(parameter_name),
        )

    missing_explicit = sorted(
        parameter_name
        for parameter_name, parameter in seed_by_name.items()
        if parameter_name not in final_inventory
        and not _is_conditional_prior_surface_parameter(parameter)
    )
    if missing_explicit:
        missing = ", ".join(missing_explicit)
        raise ValueError(
            "Stage 4 deterministic inventory drifted from compiler-exposed parameters; "
            f"compiler is missing seeded parameters: {missing}"
        )

    return _order_stage4_inventory(
        final_inventory.values(),
        retained_state_order=retained_state_order,
        retained_edges=retained_edges,
        induced_dependencies=induced_dependencies,
    )


def _fallback_initial_state_parameters(retained_state_order: list[str]) -> list[dict[str, Any]]:
    """Provide deterministic initial-state parameters when compile-time discovery is unavailable."""
    parameters: list[dict[str, Any]] = []
    for construct_name in retained_state_order:
        parameters.append(
            {
                "name": f"t0_mean_{construct_name}",
                "role": "initial_state_mean",
                "constraint": "none",
                "description": f"Initial-state mean for {construct_name}",
                "construct": construct_name,
            }
        )
    for construct_name in retained_state_order:
        parameters.append(
            {
                "name": f"t0_sd_{construct_name}",
                "role": "initial_state_sd",
                "constraint": "positive",
                "description": f"Initial-state SD for {construct_name}",
                "construct": construct_name,
            }
        )
    return parameters


def _order_stage4_inventory(
    parameters: list[dict[str, Any]] | Any,
    *,
    retained_state_order: list[str],
    retained_edges: list[dict[str, Any]],
    induced_dependencies: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return deterministically ordered parameter and loading inventories."""
    construct_order = {name: idx for idx, name in enumerate(retained_state_order)}
    edge_order = {(edge["cause"], edge["effect"]): idx for idx, edge in enumerate(retained_edges)}
    dependency_order = {
        _dependency_parameter_name(dependency): idx
        for idx, dependency in enumerate(induced_dependencies)
    }

    parameters_list = [dict(parameter) for parameter in parameters]
    role_buckets: dict[str, list[dict[str, Any]]] = {}
    loading_params: list[dict[str, Any]] = []

    for parameter in parameters_list:
        role = str(parameter["role"])
        if role == "loading":
            loading_params.append(parameter)
            continue
        role_buckets.setdefault(role, []).append(parameter)

    def _construct_key(parameter: dict[str, Any]) -> tuple[int, str]:
        construct_names = tuple(parameter.get("construct_names") or ())
        construct_name = str(
            parameter.get("construct") or (construct_names[-1] if construct_names else "")
        )
        return (construct_order.get(construct_name, len(construct_order)), construct_name)

    def _measurement_error_key(parameter: dict[str, Any]) -> tuple[int, str, str]:
        construct_name = str(parameter.get("construct") or "")
        indicator_name = str(parameter.get("indicator") or "")
        return (
            construct_order.get(construct_name, len(construct_order)),
            construct_name,
            indicator_name,
        )

    def _observation_parameter_key(parameter: dict[str, Any]) -> tuple[int, str]:
        construct_names = tuple(parameter.get("construct_names") or ())
        first_construct = str(construct_names[0]) if construct_names else ""
        return (construct_order.get(first_construct, len(construct_order)), str(parameter["name"]))

    ordered_parameters: list[dict[str, Any]] = []
    ordered_parameters.extend(
        sorted(role_buckets.pop("measurement_error_sd", []), key=_measurement_error_key)
    )
    ordered_parameters.extend(
        sorted(role_buckets.pop("observation_intercept", []), key=_measurement_error_key)
    )
    ordered_parameters.extend(
        sorted(role_buckets.pop("observation_hyperparameter", []), key=_observation_parameter_key)
    )
    ordered_parameters.extend(
        sorted(
            role_buckets.pop("observation_hyperparameter_positive", []),
            key=_observation_parameter_key,
        )
    )
    ordered_parameters.extend(sorted(role_buckets.pop("ar_coefficient", []), key=_construct_key))
    ordered_parameters.extend(
        sorted(
            role_buckets.pop("fixed_effect", []),
            key=lambda parameter: (
                edge_order.get(
                    (str(parameter.get("cause") or ""), str(parameter.get("effect") or "")),
                    len(edge_order),
                ),
                str(parameter["name"]),
            ),
        )
    )
    ordered_parameters.extend(sorted(role_buckets.pop("residual_sd", []), key=_construct_key))
    ordered_parameters.extend(sorted(role_buckets.pop("state_intercept", []), key=_construct_key))
    ordered_parameters.extend(
        sorted(role_buckets.pop("dynamics_parameter", []), key=_construct_key)
    )
    ordered_parameters.extend(
        sorted(role_buckets.pop("dynamics_parameter_positive", []), key=_construct_key)
    )
    ordered_parameters.extend(
        sorted(role_buckets.pop("initial_state_mean", []), key=_construct_key)
    )
    ordered_parameters.extend(sorted(role_buckets.pop("initial_state_sd", []), key=_construct_key))
    ordered_parameters.extend(
        sorted(
            [
                *role_buckets.pop("static_state_sd", []),
                *role_buckets.pop("correlation", []),
                *role_buckets.pop("initial_state_correlation", []),
            ],
            key=lambda parameter: (
                dependency_order.get(str(parameter["name"]), len(dependency_order)),
                str(parameter["name"]),
            ),
        )
    )
    if role_buckets:
        unknown_roles = ", ".join(sorted(role_buckets))
        raise ValueError(
            f"Unsupported Stage 4 parameter roles in deterministic ordering: {unknown_roles}"
        )

    loading_params.sort(
        key=lambda parameter: (
            construct_order.get(str(parameter.get("construct") or ""), len(construct_order)),
            str(parameter.get("indicator") or ""),
            str(parameter["name"]),
        )
    )
    return ordered_parameters, loading_params


def _dependency_parameter_name(dependency: dict[str, Any]) -> str:
    """Return the semantic Stage 4 parameter name for one induced dependency."""
    construct_1, construct_2 = dependency["between"]
    if dependency["kind"] == "innovation_correlation":
        return f"cor_{construct_1}_{construct_2}"
    return f"cor0_{construct_1}_{construct_2}"


def _is_compiler_default_only_parameter_name(parameter_name: str) -> bool:
    """Return whether a compiler-emitted name should stay hidden from Stage 4."""
    return parameter_name == "proc_df"


def _is_conditional_prior_surface_parameter(parameter: dict[str, Any]) -> bool:
    """Whether a parameter is conditional on the locked likelihood choices."""
    return bool(parameter.get("conditional_prior_surface"))


def _enrich_parameter_with_binding(
    parameter: dict[str, Any],
    binding: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach compiler binding metadata used by Stage 4 prior surfaces."""
    enriched = dict(parameter)
    if not binding:
        return enriched

    construct_names = tuple(
        name for name in binding.get("construct_names", ()) if isinstance(name, str)
    )
    indicator_names = tuple(
        name for name in binding.get("indicator_names", ()) if isinstance(name, str)
    )
    if construct_names and not enriched.get("construct_names"):
        enriched["construct_names"] = list(construct_names)
    if indicator_names and not enriched.get("indicator_names"):
        enriched["indicator_names"] = list(indicator_names)
    if construct_names and not enriched.get("construct"):
        enriched["construct"] = construct_names[-1]
    if len(construct_names) >= 2:
        enriched.setdefault("cause", construct_names[0])
        enriched.setdefault("effect", construct_names[-1])

    site_kind = binding.get("site_kind")
    prior_field = binding.get("prior_field")
    enriched["compiled_site_name"] = binding.get("site_name")
    enriched["compiled_prior_field"] = prior_field
    enriched["compiled_flat_index"] = binding.get("flat_index")
    enriched["compiled_site_kind"] = site_kind
    enriched["prior_transform"] = binding.get("transform")
    enriched["component_index"] = binding.get("component_index")
    enriched["component_parameter"] = _component_parameter_label(site_kind, prior_field)
    return enriched


def _component_parameter_label(site_kind: Any, prior_field: Any) -> str | None:
    labels = {
        "dynamics_decay": "decay",
        "dynamics_cint": "cint",
        "dynamics_weight": "weight",
        "hill_emax": "Emax",
        "hill_ec50": "EC50",
        "hill_n": "n",
    }
    key = str(site_kind or prior_field or "")
    return labels.get(key)


def _measurement_error_parameters(
    indicators: list[dict[str, Any]],
    *,
    retained_construct_names: set[str],
    indicators_per_construct: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Return one semantic measurement-error prior per free manifest channel.

    Emitted as a conditional surface: it activates only when the indicator's
    locked observation family actually reads per-channel manifest noise
    (Gaussian or Student-t). For Poisson, Gamma, Negative-Binomial, Beta,
    Bernoulli, Ordered-Logistic, or Categorical channels the emission log-prob
    ignores R, so the parameter is filtered out by
    ``parameter_is_active_for_model_spec`` before reaching authored priors.
    """
    noise_families = sorted(
        family.value for family in DistributionFamily if family.uses_manifest_noise
    )
    parameters: list[dict[str, Any]] = []
    for indicator in indicators:
        construct_name = indicator.get("construct_name")
        indicator_name = indicator["name"]
        if (
            not isinstance(construct_name, str)
            or construct_name not in retained_construct_names
            or len(indicators_per_construct.get(construct_name, ())) <= 1
        ):
            continue
        parameters.append(
            {
                "name": f"obs_sd_{indicator_name}",
                "role": "measurement_error_sd",
                "constraint": "positive",
                "description": f"Measurement-error SD for {indicator_name}",
                "construct": construct_name,
                "indicator": indicator_name,
                "activation_indicator_names": [indicator_name],
                "activation_distribution_families": list(noise_families),
                "conditional_prior_surface": True,
            }
        )
    return parameters


def _candidate_state_intercept_parameters(
    retained_constructs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return conditional continuous-time intercept surfaces for dynamic states."""
    parameters: list[dict[str, Any]] = []
    for construct in retained_constructs:
        if construct.get("temporal_status") == "time_invariant":
            continue
        construct_name = str(construct["name"])
        parameters.append(
            {
                "name": f"cint_{construct_name}",
                "role": "state_intercept",
                "constraint": "none",
                "description": f"Continuous-time state intercept for {construct_name}",
                "construct": construct_name,
                "temporal_status": construct.get("temporal_status"),
                "conditional_prior_surface": True,
                "activation_equilibrium_forcing": True,
            }
        )
    return parameters


def _candidate_initial_state_parameters(
    retained_constructs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return conditional initial-state surfaces gated by initialization policy."""
    parameters: list[dict[str, Any]] = []
    for construct in retained_constructs:
        construct_name = str(construct["name"])
        temporal_status = construct.get("temporal_status")
        parameters.append(
            {
                "name": f"t0_mean_{construct_name}",
                "role": "initial_state_mean",
                "constraint": "none",
                "description": f"Initial-state mean for {construct_name}",
                "construct": construct_name,
                "temporal_status": temporal_status,
                "conditional_prior_surface": True,
                "activation_initialization_policies": ["free"],
            }
        )
        parameters.append(
            {
                "name": f"t0_sd_{construct_name}",
                "role": "initial_state_sd",
                "constraint": "positive",
                "description": f"Initial-state SD for {construct_name}",
                "construct": construct_name,
                "temporal_status": temporal_status,
                "conditional_prior_surface": True,
                "activation_initialization_policies": ["free"],
            }
        )
    return parameters


def _candidate_observation_intercept_parameters(
    indicators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-indicator conditional manifest-intercept surfaces."""
    parameters: list[dict[str, Any]] = []
    for indicator in indicators:
        indicator_name = str(indicator["name"])
        construct_name = indicator.get("construct_name")
        parameters.append(
            {
                "name": f"manifest_mean_{indicator_name}",
                "role": "observation_intercept",
                "constraint": "none",
                "description": f"Observation intercept for {indicator_name}",
                "indicator": indicator_name,
                "construct": construct_name,
                "conditional_prior_surface": True,
            }
        )
    return parameters


def _confounder_baseline_factor_parameters(
    marginalized_scales: list[dict[str, Any]],
    *,
    retained_state_order: list[str],
) -> list[dict[str, Any]]:
    """Return one baseline-factor scale per identifiable confounder equivalence class.

    Confounders sharing the same loading column (same set of retained children)
    contribute only to ``Σ τ_c²`` in the induced covariance — their individual
    scales are not separately identifiable from data. Emit one scale per class
    and list the aggregated source confounders for elicitation context.
    """
    construct_order = {name: idx for idx, name in enumerate(retained_state_order)}
    parameters: list[dict[str, Any]] = []
    for scale in marginalized_scales:
        if scale["kind"] != "initial_state_correlation":
            continue
        affected_states = scale["affected_states"]
        if len(affected_states) < 2:
            continue
        ordered_construct_names = sorted(
            affected_states,
            key=lambda name: (construct_order.get(name, len(construct_order)), name),
        )
        sources = list(scale["sources"])
        description_sources = ", ".join(sources)
        parameters.append(
            {
                "name": scale["parameter"],
                "role": "static_state_sd",
                "constraint": "positive",
                "description": (
                    "Baseline-factor SD aggregating time-invariant confounders "
                    f"({description_sources}) that share the induced loading on "
                    f"{', '.join(ordered_construct_names)}"
                ),
                "construct_names": ordered_construct_names,
                "source_confounders": sources,
                "dependency_kind": "initial_state_correlation",
            }
        )
    return parameters


def _candidate_observation_extra_parameters(
    indicators: list[dict[str, Any]],
    *,
    resolved_likelihoods: list[dict[str, Any]],
    ambiguous_indicators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return likelihood-extra prior candidates activated by the locked family choices."""
    indicator_lookup = {indicator["name"]: indicator for indicator in indicators}
    possible_distributions_by_indicator: dict[str, set[str]] = {}

    for likelihood in resolved_likelihoods:
        variable = str(likelihood["variable"])
        possible_distributions_by_indicator.setdefault(variable, set()).add(
            str(likelihood["distribution"])
        )
    for item in ambiguous_indicators:
        variable = str(item["variable"])
        if "fixed_distribution" in item:
            possible_distributions_by_indicator.setdefault(variable, set()).add(
                str(item["fixed_distribution"])
            )
        else:
            possible_distributions_by_indicator.setdefault(variable, set()).update(
                str(distribution) for distribution in item.get("valid_distributions", ())
            )

    def _construct_names(indicator_names: list[str]) -> list[str]:
        seen: list[str] = []
        for indicator_name in indicator_names:
            construct_name = (indicator_lookup.get(indicator_name) or {}).get("construct_name")
            if isinstance(construct_name, str) and construct_name not in seen:
                seen.append(construct_name)
        return seen

    def _candidate_variables(family: DistributionFamily) -> list[str]:
        return sorted(
            indicator_name
            for indicator_name, families in possible_distributions_by_indicator.items()
            if family.value in families
        )

    candidates: list[dict[str, Any]] = []

    positive_sites = {
        "obs_df": (
            DistributionFamily.STUDENT_T,
            "Student-t observation degrees of freedom",
        ),
        "obs_shape": (
            DistributionFamily.GAMMA,
            "Gamma observation shape",
        ),
        "obs_r": (
            DistributionFamily.NEGATIVE_BINOMIAL,
            "Negative-binomial observation dispersion",
        ),
        "obs_concentration": (
            DistributionFamily.BETA,
            "Beta observation concentration",
        ),
    }
    for parameter_name, (family, description) in positive_sites.items():
        indicator_names = _candidate_variables(family)
        if not indicator_names:
            continue
        candidates.append(
            {
                "name": parameter_name,
                "role": "observation_hyperparameter_positive",
                "constraint": "positive",
                "description": description,
                "indicator_names": indicator_names,
                "construct_names": _construct_names(indicator_names),
                "activation_indicator_names": list(indicator_names),
                "activation_distribution_families": [family.value],
                "conditional_prior_surface": True,
            }
        )

    ordered_indicator_names = _candidate_variables(DistributionFamily.ORDERED_LOGISTIC)
    if ordered_indicator_names:
        candidates.append(
            {
                "name": "obs_ordered_base",
                "role": "observation_hyperparameter",
                "constraint": "none",
                "description": "Ordered-logistic threshold base locations",
                "indicator_names": ordered_indicator_names,
                "construct_names": _construct_names(ordered_indicator_names),
                "activation_indicator_names": list(ordered_indicator_names),
                "activation_distribution_families": [DistributionFamily.ORDERED_LOGISTIC.value],
                "conditional_prior_surface": True,
            }
        )

    ordered_gap_indicator_names = sorted(
        indicator_name
        for indicator_name in ordered_indicator_names
        if len((indicator_lookup.get(indicator_name) or {}).get("ordinal_levels") or ()) > 2
    )
    if ordered_gap_indicator_names:
        candidates.append(
            {
                "name": "obs_ordered_gaps",
                "role": "observation_hyperparameter_positive",
                "constraint": "positive",
                "description": "Ordered-logistic threshold gaps",
                "indicator_names": ordered_indicator_names,
                "construct_names": _construct_names(ordered_indicator_names),
                "activation_indicator_names": ordered_gap_indicator_names,
                "activation_distribution_families": [DistributionFamily.ORDERED_LOGISTIC.value],
                "conditional_prior_surface": True,
            }
        )

    categorical_indicator_names = _candidate_variables(DistributionFamily.CATEGORICAL)
    if categorical_indicator_names:
        for parameter_name, description in (
            ("obs_cat_intercepts", "Categorical class intercepts"),
            ("obs_cat_slopes", "Categorical class slopes"),
        ):
            candidates.append(
                {
                    "name": parameter_name,
                    "role": "observation_hyperparameter",
                    "constraint": "none",
                    "description": description,
                    "indicator_names": categorical_indicator_names,
                    "construct_names": _construct_names(categorical_indicator_names),
                    "activation_indicator_names": list(categorical_indicator_names),
                    "activation_distribution_families": [DistributionFamily.CATEGORICAL.value],
                    "conditional_prior_surface": True,
                }
            )

    return candidates


def _provisional_likelihood_choices(
    ambiguous_indicators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose deterministic provisional likelihoods for compiler-owned prior discovery."""
    choices: list[dict[str, Any]] = []
    for item in ambiguous_indicators:
        variable = str(item["variable"])
        if "fixed_distribution" in item:
            distribution = str(item["fixed_distribution"])
            valid_links = list(item.get("valid_links") or [])
            if not valid_links:
                raise ValueError(f"Ambiguous indicator {variable!r} is missing valid links")
            link = str(valid_links[0])
        else:
            valid_distributions = list(item.get("valid_distributions") or [])
            if not valid_distributions:
                raise ValueError(f"Ambiguous indicator {variable!r} is missing valid distributions")
            distribution = str(valid_distributions[0])
            link_options = item.get("link_options") or {}
            valid_links = list(link_options.get(distribution) or [])
            if not valid_links:
                raise ValueError(
                    f"Ambiguous indicator {variable!r} is missing link options for {distribution!r}"
                )
            link = str(valid_links[0])
        choices.append(
            {
                "variable": variable,
                "construct_name": item.get("construct_name"),
                "distribution": distribution,
                "link": link,
                "support_kind": item.get("support_kind"),
                "summary_operator": item.get("summary_operator"),
                "centered": False,
                "reasoning": "Deterministic provisional choice for compiler-owned prior discovery.",
            }
        )
    return choices


def _parameter_metadata_from_compiler_row(
    parameter_name: str,
    *,
    binding: dict[str, Any] | None,
    retained_construct_names: set[str],
) -> dict[str, Any] | None:
    """Convert one compiler-owned extra prior row into Stage 4 parameter metadata."""
    if parameter_name.startswith("t0_mean_"):
        construct_name = parameter_name.removeprefix("t0_mean_")
        if construct_name in retained_construct_names:
            return {
                "name": parameter_name,
                "role": "initial_state_mean",
                "constraint": "none",
                "description": f"Initial-state mean for {construct_name}",
                "construct": construct_name,
            }
        return None

    if parameter_name.startswith("t0_sd_"):
        construct_name = parameter_name.removeprefix("t0_sd_")
        if construct_name in retained_construct_names:
            return {
                "name": parameter_name,
                "role": "initial_state_sd",
                "constraint": "positive",
                "description": f"Initial-state SD for {construct_name}",
                "construct": construct_name,
            }
        return None

    if binding is not None and _is_component_owned_dynamics_binding(binding):
        construct_names = tuple(
            name for name in binding.get("construct_names", ()) if isinstance(name, str)
        )
        positive = str(binding.get("transform") or "") == "positive_identity" or str(
            binding.get("site_kind") or ""
        ) in {"dynamics_decay", "hill_emax", "hill_ec50"}
        return {
            "name": parameter_name,
            "role": "dynamics_parameter_positive" if positive else "dynamics_parameter",
            "constraint": "positive" if positive else "none",
            "description": f"Component dynamics parameter {parameter_name}",
            "construct_names": list(construct_names),
        }

    return None


def _is_component_owned_dynamics_binding(binding: dict[str, Any]) -> bool:
    return str(binding.get("site_kind") or "") in {
        "dynamics_decay",
        "dynamics_cint",
        "dynamics_weight",
        "hill_emax",
        "hill_ec50",
        "hill_n",
    }
