"""Pure spec-translation stage for the SSM compilation pipeline."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.models.likelihoods.observation_families import supported_distribution_families
from causal_ssm_agent.models.ssm.model import SSMSpec
from causal_ssm_agent.orchestrator.schemas import parse_duration_to_hours
from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionFamily,
    LinkFunction,
    ModelSpec,
    ParameterRole,
)
from causal_ssm_agent.utils.causal_spec import get_constructs, get_edges, get_indicators


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


def get_structural_latent_layout(
    causal_spec: dict | None,
) -> tuple[list[str], np.ndarray | None] | None:
    """Build the canonical latent ordering from the causal structure."""
    if causal_spec is None:
        return None

    constructs = get_constructs(causal_spec)
    if not constructs:
        raise ValueError("causal_spec.latent.constructs is empty")

    time_varying: list[str] = []
    time_invariant: list[str] = []
    seen: set[str] = set()

    for construct in constructs:
        name = construct.get("name") if isinstance(construct, dict) else construct.name
        temporal = (
            construct.get("temporal_status")
            if isinstance(construct, dict)
            else construct.temporal_status
        )
        if name in seen:
            raise ValueError(f"Duplicate construct name in causal_spec: {name!r}")
        seen.add(name)
        if temporal == "time_invariant":
            time_invariant.append(name)
        else:
            time_varying.append(name)

    latent_names = time_varying + time_invariant
    time_invariant_mask = None
    if time_invariant:
        time_invariant_mask = np.array(
            [False] * len(time_varying) + [True] * len(time_invariant),
            dtype=bool,
        )
    return latent_names, time_invariant_mask


def build_masks_from_causal_spec(
    latent_names: list[str] | None,
    manifest_cols: list[str],
    n_latent: int,
    n_manifest: int,
    *,
    causal_spec: dict | None,
) -> tuple[np.ndarray | None, jnp.ndarray, np.ndarray | None, dict[tuple[int, int], float]]:
    """Build drift/lambda masks and edge lag metadata from the causal structure."""
    if causal_spec is None or latent_names is None:
        return None, jnp.eye(n_manifest, n_latent), None, {}

    edges = get_edges(causal_spec)
    indicators = get_indicators(causal_spec)

    indicator_names = {
        (indicator.get("name") if isinstance(indicator, dict) else indicator.name)
        for indicator in indicators
    }
    unknown_likelihoods = sorted(set(manifest_cols) - indicator_names)
    if unknown_likelihoods:
        raise ValueError(
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
    reference_set: set[str] = set()
    matched_manifests: set[str] = set()

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
            raise ValueError(
                "CausalSpec measurement indicator references unknown construct: "
                f"{ind_name!r} -> {construct_name!r}"
            )

        manifest_idx_value = manifest_idx[ind_name]
        latent_idx_value = latent_idx[construct_name]
        matched_manifests.add(ind_name)

        if construct_name not in reference_set:
            lambda_mat_np[manifest_idx_value, latent_idx_value] = 1.0
            reference_set.add(construct_name)
        else:
            lambda_mask[manifest_idx_value, latent_idx_value] = True

    lambda_mat = jnp.array(lambda_mat_np)

    unmatched_manifests = sorted(set(manifest_cols) - matched_manifests)
    if unmatched_manifests:
        raise ValueError(
            "ModelSpec likelihoods could not be mapped to causal_spec measurement indicators: "
            f"{unmatched_manifests}"
        )

    return drift_mask, lambda_mat, lambda_mask, edge_lag_days


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

    structural_layout = get_structural_latent_layout(causal_spec)
    if structural_layout is not None:
        latent_names, time_invariant_mask = structural_layout
        n_latent = len(latent_names)
    else:
        ar_params = [
            param for param in model_spec.parameters if param.role == ParameterRole.AR_COEFFICIENT
        ]
        if not ar_params:
            raise ValueError(
                "No AR_COEFFICIENT parameters found in ModelSpec; "
                "cannot infer latent dimensionality without causal_spec."
            )
        n_latent = len(ar_params)
        latent_names = [param.name.removeprefix("rho_") for param in ar_params]
        time_invariant_mask = None

    manifest_dists: list[DistributionFamily] = []
    supported_families = supported_distribution_families()
    for likelihood in model_spec.likelihoods:
        dist = likelihood.distribution
        if dist not in supported_families:
            supported = sorted(distribution.value for distribution in supported_families)
            raise ValueError(
                f"Indicator '{likelihood.variable}': distribution '{dist}' "
                f"has no native emission function. Supported: {supported}."
            )
        manifest_dists.append(dist)

    manifest_dist = DistributionFamily.GAUSSIAN
    for dist in manifest_dists:
        if dist != DistributionFamily.GAUSSIAN:
            manifest_dist = dist
            break

    manifest_links: list[LinkFunction] = [likelihood.link for likelihood in model_spec.likelihoods]
    manifest_link = LinkFunction.IDENTITY
    for link in manifest_links:
        if link != LinkFunction.IDENTITY:
            manifest_link = link
            break

    drift_mask, lambda_mat, lambda_mask, edge_lag_days = build_masks_from_causal_spec(
        latent_names,
        manifest_cols,
        n_latent,
        n_manifest,
        causal_spec=causal_spec,
    )

    has_correlation = any(
        parameter.role == ParameterRole.CORRELATION for parameter in model_spec.parameters
    )
    diffusion_mode = "free" if has_correlation else "diag"

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        lambda_mat=lambda_mat,
        drift="free",
        diffusion=diffusion_mode,
        cint="free",
        manifest_means=None,
        manifest_var="diag",
        manifest_dist=manifest_dist,
        manifest_dists=manifest_dists,
        manifest_link=manifest_link,
        manifest_links=manifest_links,
        t0_means="free",
        t0_var="diag",
        latent_names=latent_names,
        manifest_names=manifest_cols,
        drift_mask=drift_mask,
        lambda_mask=lambda_mask,
        time_invariant_mask=time_invariant_mask,
    )
    return spec, edge_lag_days
