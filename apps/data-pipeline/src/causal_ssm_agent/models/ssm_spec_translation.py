"""Pure spec-translation stage for the SSM compilation pipeline."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.compilation_errors import AggregatedCompileError
from causal_ssm_agent.models.ssm.inference.targets.observation_families import (
    supported_distribution_families,
)
from causal_ssm_agent.models.ssm.model import SSMSpec, full_drift_mask, zero_loading_mask
from causal_ssm_agent.models.ssm.parameter_names import build_initial_state_correlation_mask
from causal_ssm_agent.orchestrator.schemas import parse_duration_to_hours
from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionFamily,
    LinkFunction,
    ModelSpec,
    ParameterRole,
)
from causal_ssm_agent.utils.causal_spec import (
    build_reference_indicator_lookup,
    get_estimation_edges,
    get_estimation_state_order,
    get_indicator_polarity,
    get_indicators,
    get_latent_constructs,
)


class SpecTranslationError(AggregatedCompileError):
    """Aggregate independent ``ModelSpec`` -> ``SSMSpec`` translation errors."""

    header = "Spec translation failed"


def get_construct_dt_days(causal_spec: dict | None, _construct_name: str = "") -> float:
    """Get the model clock interval in fractional days."""
    if causal_spec is None:
        return 1.0

    model_clock = (
        causal_spec.get("measurement", {}).get("model_clock")
        if isinstance(causal_spec, dict)
        else getattr(getattr(causal_spec, "measurement", None), "model_clock", None)
    )
    if not model_clock:
        return 1.0

    try:
        return parse_duration_to_hours(model_clock) / 24.0
    except ValueError:
        return 1.0


def get_estimation_latent_layout(
    causal_spec: dict | None,
) -> tuple[list[str], np.ndarray | None] | None:
    """Build the canonical latent ordering from the retained estimation states."""
    if causal_spec is None:
        return None

    try:
        state_order = get_estimation_state_order(causal_spec)
    except ValueError as exc:
        raise SpecTranslationError([str(exc)]) from exc
    errors: list[str] = []
    if not state_order:
        errors.append("causal_spec.estimation.state_order is empty")
        raise SpecTranslationError(errors)

    latent_construct_lookup = {
        construct["name"]: construct for construct in get_latent_constructs(causal_spec)
    }
    unknown_states = [name for name in state_order if name not in latent_construct_lookup]
    if unknown_states:
        errors.append(
            "causal_spec.estimation.state_order references constructs absent from latent.constructs: "
            f"{sorted(unknown_states)}"
        )
        raise SpecTranslationError(errors)

    time_invariant_mask = np.array(
        [
            latent_construct_lookup[name].get("temporal_status") == "time_invariant"
            for name in state_order
        ],
        dtype=bool,
    )
    if not bool(time_invariant_mask.any()):
        time_invariant_mask = None
    return state_order, time_invariant_mask


def build_masks_from_causal_spec(
    latent_names: list[str] | None,
    manifest_cols: list[str],
    n_latent: int,
    n_manifest: int,
    *,
    causal_spec: dict | None,
) -> tuple[np.ndarray, jnp.ndarray, np.ndarray, dict[tuple[int, int], float]]:
    """Build drift/lambda masks and edge lag metadata from the causal structure."""
    if causal_spec is None or latent_names is None:
        return (
            full_drift_mask(n_latent),
            jnp.eye(n_manifest, n_latent),
            zero_loading_mask(n_manifest, n_latent),
            {},
        )

    try:
        edges = get_estimation_edges(causal_spec)
    except ValueError as exc:
        raise SpecTranslationError([str(exc)]) from exc
    indicators = get_indicators(causal_spec)
    errors: list[str] = []

    indicator_names = {
        (indicator.get("name") if isinstance(indicator, dict) else indicator.name)
        for indicator in indicators
    }
    unknown_likelihoods = sorted(set(manifest_cols) - indicator_names)
    if unknown_likelihoods:
        errors.append(
            "ModelSpec likelihoods reference indicators absent from causal_spec measurement: "
            f"{unknown_likelihoods}"
        )

    latent_idx = {name: idx for idx, name in enumerate(latent_names)}
    drift_mask = np.eye(n_latent, dtype=bool)
    edge_lag_days: dict[tuple[int, int], float] = {}
    model_dt_days = get_construct_dt_days(causal_spec)

    for edge in edges:
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if cause not in latent_idx or effect not in latent_idx:
            continue
        effect_idx, cause_idx = latent_idx[effect], latent_idx[cause]
        drift_mask[effect_idx, cause_idx] = True

        lagged = edge.get("lagged", True) if isinstance(edge, dict) else edge.lagged
        lag_hours = model_dt_days * 24.0 if lagged else 0.0
        if lag_hours > 0:
            edge_lag_days[(effect_idx, cause_idx)] = lag_hours / 24.0

    manifest_idx = {name: idx for idx, name in enumerate(manifest_cols)}
    lambda_mat_np = np.zeros((n_manifest, n_latent), dtype=np.float64)
    lambda_mask = np.zeros((n_manifest, n_latent), dtype=bool)
    reference_indicator_lookup = build_reference_indicator_lookup(indicators)
    matched_manifests: set[str] = set()
    invalid_construct_manifests: set[str] = set()

    for indicator in indicators:
        ind_name = indicator.get("name") if isinstance(indicator, dict) else indicator.name
        construct_name = (
            indicator.get("construct_name")
            if isinstance(indicator, dict)
            else indicator.construct_name
        )
        if ind_name not in manifest_idx:
            continue
        if construct_name not in latent_idx:
            errors.append(
                "CausalSpec measurement indicator references unknown construct: "
                f"{ind_name!r} -> {construct_name!r}"
            )
            invalid_construct_manifests.add(ind_name)
            continue

        manifest_idx_value = manifest_idx[ind_name]
        latent_idx_value = latent_idx[construct_name]
        matched_manifests.add(ind_name)

        if ind_name == reference_indicator_lookup.get(construct_name):
            lambda_mat_np[manifest_idx_value, latent_idx_value] = (
                1.0 if get_indicator_polarity(indicator) == "positive" else -1.0
            )
        else:
            lambda_mask[manifest_idx_value, latent_idx_value] = True

    lambda_mat = jnp.array(lambda_mat_np)

    unmatched_manifests = sorted(
        set(manifest_cols)
        - matched_manifests
        - set(unknown_likelihoods)
        - invalid_construct_manifests
    )
    if unmatched_manifests:
        errors.append(
            "ModelSpec likelihoods could not be mapped to causal_spec measurement indicators: "
            f"{unmatched_manifests}"
        )

    if errors:
        raise SpecTranslationError(errors)

    return drift_mask, lambda_mat, lambda_mask, edge_lag_days


def build_manifest_variance_from_causal_spec(
    latent_names: list[str] | None,
    manifest_cols: list[str],
    *,
    causal_spec: dict | None,
) -> tuple[jnp.ndarray | str, np.ndarray | None]:
    """Build manifest-noise structure from the retained measurement model.

    Single-indicator constructs absorb measurement error into the structural
    residual, so their manifest channels get fixed zero observation noise.
    Multi-indicator constructs keep free diagonal manifest noise.
    """
    if causal_spec is None or latent_names is None:
        return "diag", None

    indicators = get_indicators(causal_spec)
    latent_name_set = set(latent_names)
    manifest_idx = {name: idx for idx, name in enumerate(manifest_cols)}
    manifest_to_construct: dict[str, str] = {}
    indicators_per_construct: dict[str, int] = {}

    for indicator in indicators:
        ind_name = indicator.get("name") if isinstance(indicator, dict) else indicator.name
        construct_name = (
            indicator.get("construct_name")
            if isinstance(indicator, dict)
            else indicator.construct_name
        )
        if ind_name not in manifest_idx or construct_name not in latent_name_set:
            continue
        manifest_to_construct[ind_name] = construct_name
        indicators_per_construct[construct_name] = (
            indicators_per_construct.get(construct_name, 0) + 1
        )

    if not manifest_to_construct:
        return "diag", None

    manifest_var_mask = np.ones(len(manifest_cols), dtype=bool)
    fixed_any = False
    for manifest_name, construct_name in manifest_to_construct.items():
        if indicators_per_construct.get(construct_name) == 1:
            manifest_var_mask[manifest_idx[manifest_name]] = False
            fixed_any = True

    if not fixed_any:
        return "diag", None

    manifest_var = np.zeros((len(manifest_cols), len(manifest_cols)), dtype=np.float64)
    return jnp.array(manifest_var), manifest_var_mask


def build_manifest_level_counts_from_causal_spec(
    manifest_cols: list[str],
    manifest_dists: list[DistributionFamily],
    *,
    causal_spec: dict | None,
) -> list[int] | None:
    """Build per-manifest discrete level counts from causal-spec metadata."""
    if causal_spec is None:
        return None

    needs_level_metadata = any(
        dist in {DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL}
        for dist in manifest_dists
    )
    if not needs_level_metadata:
        return None

    indicator_lookup = {
        (indicator.get("name") if isinstance(indicator, dict) else indicator.name): indicator
        for indicator in get_indicators(causal_spec)
    }
    level_counts = [0] * len(manifest_cols)
    errors: list[str] = []

    for idx, (manifest_name, dist) in enumerate(zip(manifest_cols, manifest_dists, strict=False)):
        if dist != DistributionFamily.ORDERED_LOGISTIC:
            continue

        indicator = indicator_lookup.get(manifest_name)
        ordinal_levels = (
            indicator.get("ordinal_levels")
            if isinstance(indicator, dict)
            else getattr(indicator, "ordinal_levels", None)
        )
        if not ordinal_levels or len(ordinal_levels) < 2:
            errors.append(
                f"Indicator '{manifest_name}' uses ordered_logistic but causal_spec is missing "
                "ordinal_levels with at least 2 levels"
            )
            continue
        level_counts[idx] = len(ordinal_levels)

    if errors:
        raise SpecTranslationError(errors)
    return level_counts


def translate_spec(
    model_spec: ModelSpec | dict,
    causal_spec: dict | None = None,
) -> tuple[SSMSpec, dict[tuple[int, int], float]]:
    """Translate ``ModelSpec`` into ``SSMSpec`` with explicit edge-lag metadata.

    Assumes the caller has already validated ``model_spec`` (e.g. via
    ``ssm_compiler.validate_model_spec_for_compilation``).  This function is
    a pure translation stage — it does not re-validate.
    """
    if isinstance(model_spec, dict):
        model_spec = ModelSpec.model_validate(model_spec)

    manifest_cols = [lik.variable for lik in model_spec.likelihoods]
    n_manifest = len(manifest_cols)
    errors: list[str] = []

    layout_failed = False
    try:
        structural_layout = get_estimation_latent_layout(causal_spec)
    except SpecTranslationError as exc:
        structural_layout = None
        layout_failed = True
        errors.extend(exc.errors)
    if structural_layout is not None:
        latent_names, time_invariant_mask = structural_layout
        n_latent = len(latent_names)
    else:
        if causal_spec is not None and layout_failed:
            latent_names = []
            time_invariant_mask = None
            n_latent = 0
        else:
            ar_params = [
                param
                for param in model_spec.parameters
                if param.role == ParameterRole.AR_COEFFICIENT
            ]
            if not ar_params:
                errors.append(
                    "No AR_COEFFICIENT parameters found in ModelSpec; "
                    "cannot infer latent dimensionality without causal_spec."
                )
                raise SpecTranslationError(errors)
            n_latent = len(ar_params)
            latent_names = [param.name.removeprefix("rho_") for param in ar_params]
            time_invariant_mask = None

    manifest_dists: list[DistributionFamily] = []
    supported_families = supported_distribution_families()
    for likelihood in model_spec.likelihoods:
        dist = likelihood.distribution
        if dist not in supported_families:
            supported = sorted(distribution.value for distribution in supported_families)
            errors.append(
                f"Indicator '{likelihood.variable}': distribution '{dist}' "
                f"has no native emission function. Supported: {supported}."
            )
        manifest_dists.append(dist)

    if causal_spec is not None and layout_failed:
        raise SpecTranslationError(errors)

    manifest_links: list[LinkFunction] = [likelihood.link for likelihood in model_spec.likelihoods]

    try:
        drift_mask, lambda_mat, lambda_mask, edge_lag_days = build_masks_from_causal_spec(
            latent_names,
            manifest_cols,
            n_latent,
            n_manifest,
            causal_spec=causal_spec,
        )
    except SpecTranslationError as exc:
        errors.extend(exc.errors)
        drift_mask = full_drift_mask(n_latent)
        lambda_mat = jnp.eye(n_manifest, n_latent)
        lambda_mask = zero_loading_mask(n_manifest, n_latent)
        edge_lag_days = {}

    manifest_var, manifest_var_mask = build_manifest_variance_from_causal_spec(
        latent_names,
        manifest_cols,
        causal_spec=causal_spec,
    )
    try:
        manifest_level_counts = build_manifest_level_counts_from_causal_spec(
            manifest_cols,
            manifest_dists,
            causal_spec=causal_spec,
        )
    except SpecTranslationError as exc:
        errors.extend(exc.errors)
        manifest_level_counts = None

    has_innovation_correlation = any(
        parameter.role == ParameterRole.CORRELATION for parameter in model_spec.parameters
    )
    try:
        t0_correlation_mask = build_initial_state_correlation_mask(latent_names, model_spec)
    except ValueError as exc:
        errors.append(str(exc))
        t0_correlation_mask = None
    diffusion_mode = "free" if has_innovation_correlation else "diag"
    t0_var_mode = "free" if t0_correlation_mask is not None else "diag"

    if errors:
        raise SpecTranslationError(errors)

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        lambda_mat=lambda_mat,
        drift="free",
        diffusion=diffusion_mode,
        diffusion_dists=[DistributionFamily.GAUSSIAN] * n_latent,
        cint="free",
        manifest_means=None,
        manifest_var=manifest_var,
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
        manifest_level_counts=manifest_level_counts,
        t0_means="free",
        t0_var=t0_var_mode,
        latent_names=latent_names,
        manifest_names=manifest_cols,
        drift_mask=drift_mask,
        lambda_mask=lambda_mask,
        manifest_var_mask=manifest_var_mask,
        t0_correlation_mask=t0_correlation_mask,
        time_invariant_mask=time_invariant_mask,
    )
    return spec, edge_lag_days
