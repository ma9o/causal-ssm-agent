"""Pure compilation pipeline for turning ModelSpec + priors into SSM inputs."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.likelihoods.base import NUMERICAL_EPSILON
from causal_ssm_agent.models.ssm.model import SSMPriors, SSMSpec
from causal_ssm_agent.orchestrator.schemas import parse_duration_to_hours
from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionFamily,
    LinkFunction,
    ModelSpec,
    ParameterRole,
)
from causal_ssm_agent.utils.causal_spec import get_constructs, get_edges, get_indicators

logger = get_prefect_logger(__name__)

PriorIndexMaps = tuple[
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
]

SUPPORTED_EMISSIONS: set[DistributionFamily] = {
    DistributionFamily.GAUSSIAN,
    DistributionFamily.STUDENT_T,
    DistributionFamily.POISSON,
    DistributionFamily.GAMMA,
    DistributionFamily.BERNOULLI,
    DistributionFamily.NEGATIVE_BINOMIAL,
    DistributionFamily.BETA,
    DistributionFamily.ORDERED_LOGISTIC,
    DistributionFamily.CATEGORICAL,
}

ROLE_TO_SSM: dict[ParameterRole, tuple[str, dict[str, float]]] = {
    ParameterRole.AR_COEFFICIENT: ("drift_diag", {"mu": -0.5, "sigma": 1.0}),
    ParameterRole.FIXED_EFFECT: ("drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    ParameterRole.RESIDUAL_SD: ("diffusion_diag", {"sigma": 1.0}),
    ParameterRole.LOADING: ("lambda_free", {"mu": 0.5, "sigma": 0.5}),
    ParameterRole.CORRELATION: ("diffusion_offdiag", {"mu": 0.0, "sigma": 0.5}),
}

KEYWORD_RULES: list[tuple[list[str], str, dict[str, float]]] = [
    (["rho", "ar"], "drift_diag", {"mu": -0.5, "sigma": 1.0}),
    (["beta"], "drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    (["sigma", "sd"], "diffusion_diag", {"sigma": 1.0}),
    (["lambda", "loading"], "lambda_free", {"mu": 0.5, "sigma": 0.5}),
    (["cor"], "diffusion_offdiag", {"mu": 0.0, "sigma": 0.5}),
]

SAMPLE_SITE_FOR_PRIOR_FIELD: dict[str, str] = {
    "drift_diag": "drift_diag_pop",
    "drift_offdiag": "drift_offdiag_pop",
    "diffusion_diag": "diffusion_diag_pop",
    "diffusion_offdiag": "diffusion_lower",
    "lambda_free": "lambda_free",
}


def normalize_prior_params(distribution: str, params: dict) -> dict[str, float]:
    """Convert distribution-specific params to the mu/sigma shape used by SSMPriors."""
    dist_lower = distribution.lower()

    if dist_lower in {"normal", "truncatednormal"}:
        return {"mu": params.get("mu", 0.0), "sigma": params.get("sigma", 1.0)}

    if dist_lower == "halfnormal":
        return {"sigma": params.get("sigma", 1.0)}

    if dist_lower == "beta":
        alpha = params.get("alpha", 2.0)
        beta = params.get("beta", 2.0)
        mu = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return {"mu": mu, "sigma": var**0.5}

    if dist_lower == "uniform":
        lower = params.get("lower", -1.0)
        upper = params.get("upper", 1.0)
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 4
        return {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper}

    return {"mu": params.get("mu", 0.0), "sigma": params.get("sigma", 1.0)}


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


def expected_prior_size(attr: str, ssm_spec: SSMSpec | None) -> int | None:
    """Return the structural size for an array-valued prior field."""
    if ssm_spec is None:
        return None

    if attr in {"drift_diag", "diffusion_diag"}:
        return ssm_spec.n_latent

    if attr == "drift_offdiag":
        if ssm_spec.drift_mask is None:
            return ssm_spec.n_latent * (ssm_spec.n_latent - 1)
        count = 0
        for i in range(ssm_spec.n_latent):
            for j in range(ssm_spec.n_latent):
                if i != j and ssm_spec.drift_mask[i, j]:
                    count += 1
        return count

    if attr == "lambda_free":
        if ssm_spec.lambda_mask is None:
            return None
        return int(np.asarray(ssm_spec.lambda_mask).sum())

    if attr == "diffusion_offdiag":
        if ssm_spec.diffusion != "free":
            return 0
        n_latent = ssm_spec.n_latent
        return n_latent * (n_latent - 1) // 2

    return None


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
    """Translate ``ModelSpec`` into ``SSMSpec`` with explicit edge-lag metadata."""
    if isinstance(model_spec, dict):
        model_spec = ModelSpec.model_validate(model_spec)

    from causal_ssm_agent.models.ssm_compiler import validate_model_spec_for_compilation

    validated_model_spec, errors = validate_model_spec_for_compilation(
        model_spec,
        causal_spec=causal_spec,
    )
    if errors:
        raise ValueError("ModelSpec failed compiler validation:\n" + "\n".join(errors))
    assert validated_model_spec is not None
    model_spec = validated_model_spec

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
    for likelihood in model_spec.likelihoods:
        dist = likelihood.distribution
        if dist not in SUPPORTED_EMISSIONS:
            raise ValueError(
                f"Indicator '{likelihood.variable}': distribution '{dist}' "
                f"has no native emission function. Supported: {sorted(d.value for d in SUPPORTED_EMISSIONS)}."
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
        param.role == ParameterRole.CORRELATION for param in model_spec.parameters
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


def warn_first_order_approximation(ssm_priors: SSMPriors) -> None:
    """Warn when the first-order DT->CT approximation is likely inaccurate."""
    diag_prior = ssm_priors.drift_diag
    offdiag_prior = ssm_priors.drift_offdiag
    if diag_prior is None or offdiag_prior is None:
        return

    diag_mu = diag_prior.get("mu")
    offdiag_mu = offdiag_prior.get("mu")
    if diag_mu is None or offdiag_mu is None:
        return

    if isinstance(diag_mu, (int, float)):
        diag_mu = [diag_mu]
    if isinstance(offdiag_mu, (int, float)):
        offdiag_mu = [offdiag_mu]

    if not diag_mu or not offdiag_mu:
        return

    min_diag = min(abs(float(value)) for value in diag_mu)
    if min_diag < NUMERICAL_EPSILON:
        return

    for idx, offdiag_value in enumerate(offdiag_mu):
        ratio = abs(float(offdiag_value)) / min_diag
        if ratio <= 0.2:
            continue
        logger.warning(
            "First-order DT->CT approximation may be inaccurate: "
            "off-diagonal drift[%d] magnitude (%.3f) is %.0f%% of "
            "minimum diagonal magnitude (%.3f). Consider a shorter "
            "reference interval or eliciting priors directly on CT rates.",
            idx,
            abs(float(offdiag_value)),
            ratio * 100,
            min_diag,
        )
        break


def check_drift_lag_consistency(
    ssm_priors: SSMPriors,
    ssm_spec: SSMSpec,
    *,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
) -> None:
    """Check CT drift rates against expected lag metadata from the causal structure."""
    edge_lags = edge_lag_days or {}
    if not edge_lags:
        return

    offdiag_prior = ssm_priors.drift_offdiag
    if offdiag_prior is None or "mu" not in offdiag_prior:
        return

    mu_arr = offdiag_prior["mu"]
    if not isinstance(mu_arr, list):
        return

    n_latent = ssm_spec.n_latent
    offdiag_positions: list[tuple[int, int]] = []
    if ssm_spec.drift_mask is not None:
        for effect_idx in range(n_latent):
            for cause_idx in range(n_latent):
                if effect_idx != cause_idx and ssm_spec.drift_mask[effect_idx, cause_idx]:
                    offdiag_positions.append((effect_idx, cause_idx))

    for flat_idx, (effect_idx, cause_idx) in enumerate(offdiag_positions):
        if flat_idx >= len(mu_arr) or (effect_idx, cause_idx) not in edge_lags:
            continue

        mu_ct = abs(float(mu_arr[flat_idx]))
        if mu_ct < NUMERICAL_EPSILON:
            continue

        expected_lag_days = edge_lags[(effect_idx, cause_idx)]
        implied_timescale_days = 1.0 / mu_ct
        ratio = max(implied_timescale_days, expected_lag_days) / max(
            min(implied_timescale_days, expected_lag_days),
            NUMERICAL_EPSILON,
        )
        if ratio <= 5.0:
            continue

        cause_name = (
            ssm_spec.latent_names[cause_idx] if ssm_spec.latent_names else f"latent_{cause_idx}"
        )
        effect_name = (
            ssm_spec.latent_names[effect_idx] if ssm_spec.latent_names else f"latent_{effect_idx}"
        )
        logger.warning(
            "Drift rate for %s->%s implies timescale %.1f days, but edge lag suggests %.1f days "
            "(%.0fx mismatch). The literature prior may be calibrated to a different observation "
            "interval than the causal model expects.",
            cause_name,
            effect_name,
            implied_timescale_days,
            expected_lag_days,
            ratio,
        )


def build_prior_index_maps(
    ssm_spec: SSMSpec | None,
    model_spec: ModelSpec | dict | None,
    *,
    causal_spec: dict | None = None,
) -> PriorIndexMaps:
    """Build parameter-name -> (SSMPriors field, flat index) maps."""
    offdiag_index: dict[str, tuple[str, int]] = {}
    lambda_index: dict[str, tuple[str, int]] = {}
    diag_index: dict[str, tuple[str, int]] = {}
    diffusion_diag_index: dict[str, tuple[str, int]] = {}
    diffusion_offdiag_index: dict[str, tuple[str, int]] = {}

    if ssm_spec is None or not model_spec:
        return (
            offdiag_index,
            lambda_index,
            diag_index,
            diffusion_diag_index,
            diffusion_offdiag_index,
        )

    if isinstance(model_spec, dict):
        spec_obj = ModelSpec.model_validate(model_spec)
    elif isinstance(model_spec, ModelSpec):
        spec_obj = model_spec
    else:
        return (
            offdiag_index,
            lambda_index,
            diag_index,
            diffusion_diag_index,
            diffusion_offdiag_index,
        )

    latent_names = ssm_spec.latent_names or []
    latent_idx_map = {name: idx for idx, name in enumerate(latent_names)}
    latent_name_set = set(latent_idx_map)
    strict_structure = causal_spec is not None

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.AR_COEFFICIENT:
            continue
        construct = parameter.name.removeprefix("rho_").removeprefix("ar_")
        if construct in latent_idx_map:
            diag_index[parameter.name] = ("drift_diag", latent_idx_map[construct])
        elif strict_structure:
            raise ValueError(
                "AR parameter does not reference a construct in causal_spec: "
                f"{parameter.name!r} not in {sorted(latent_idx_map)}"
            )

    for parameter in spec_obj.parameters:
        if parameter.role != ParameterRole.RESIDUAL_SD:
            continue
        construct = parameter.name.removeprefix("sigma_")
        if construct in latent_idx_map:
            diffusion_diag_index[parameter.name] = ("diffusion_diag", latent_idx_map[construct])
        elif strict_structure:
            raise ValueError(
                "RESIDUAL_SD parameter does not reference a construct in causal_spec: "
                f"{parameter.name!r} not in {sorted(latent_idx_map)}"
            )

    if ssm_spec.drift_mask is not None:
        positions: list[tuple[int, int]] = []
        for effect_idx in range(ssm_spec.n_latent):
            for cause_idx in range(ssm_spec.n_latent):
                if effect_idx != cause_idx and ssm_spec.drift_mask[effect_idx, cause_idx]:
                    positions.append((effect_idx, cause_idx))

        for parameter in spec_obj.parameters:
            if parameter.role != ParameterRole.FIXED_EFFECT:
                continue
            compound = parameter.name.removeprefix("beta_")
            result = split_compound_name(compound, latent_name_set, latent_name_set)
            if result is None:
                message = (
                    "Could not parse FIXED_EFFECT parameter "
                    f"{parameter.name!r} into (cause, effect) from known latents {sorted(latent_name_set)}"
                )
                if strict_structure:
                    raise ValueError(message)
                logger.warning("%s", message)
                continue
            cause_name, effect_name = result
            position = (latent_idx_map[effect_name], latent_idx_map[cause_name])
            if position in positions:
                offdiag_index[parameter.name] = ("drift_offdiag", positions.index(position))
            elif strict_structure:
                raise ValueError(
                    "FIXED_EFFECT parameter does not correspond to an edge in causal_spec: "
                    f"{parameter.name!r}"
                )

    if ssm_spec.lambda_mask is not None:
        manifest_names = ssm_spec.manifest_names or []
        manifest_idx_map = {name: idx for idx, name in enumerate(manifest_names)}
        manifest_name_set = set(manifest_idx_map)

        positions: list[tuple[int, int]] = []
        for manifest_idx in range(ssm_spec.n_manifest):
            for latent_idx in range(ssm_spec.n_latent):
                if ssm_spec.lambda_mask[manifest_idx, latent_idx]:
                    positions.append((manifest_idx, latent_idx))

        for parameter in spec_obj.parameters:
            if parameter.role != ParameterRole.LOADING:
                continue
            compound = parameter.name.removeprefix("lambda_")
            result = split_compound_name(compound, manifest_name_set, latent_name_set)
            if result is None:
                message = (
                    "Could not parse LOADING parameter "
                    f"{parameter.name!r} into (indicator, construct) from known manifests "
                    f"{sorted(manifest_name_set)} / latents {sorted(latent_name_set)}"
                )
                if strict_structure:
                    raise ValueError(message)
                logger.warning("%s", message)
                continue
            indicator_name, construct_name = result
            position = (manifest_idx_map[indicator_name], latent_idx_map[construct_name])
            if position in positions:
                lambda_index[parameter.name] = ("lambda_free", positions.index(position))
            elif strict_structure:
                raise ValueError(
                    "LOADING parameter does not correspond to a free loading in causal_spec: "
                    f"{parameter.name!r}"
                )

    if ssm_spec.diffusion == "free":
        lower_positions: list[tuple[int, int]] = []
        for i in range(ssm_spec.n_latent):
            for j in range(i):
                lower_positions.append((i, j))

        for parameter in spec_obj.parameters:
            if parameter.role != ParameterRole.CORRELATION:
                continue
            compound = parameter.name.removeprefix("cor_")
            result = split_compound_name(compound, latent_name_set, latent_name_set)
            if result is None:
                message = (
                    "Could not parse CORRELATION parameter "
                    f"{parameter.name!r} into (state1, state2) from known latents {sorted(latent_name_set)}"
                )
                if strict_structure:
                    raise ValueError(message)
                logger.warning("%s", message)
                continue
            state1_name, state2_name = result
            idx1 = latent_idx_map[state1_name]
            idx2 = latent_idx_map[state2_name]
            position = (max(idx1, idx2), min(idx1, idx2))
            if position in lower_positions:
                diffusion_offdiag_index[parameter.name] = (
                    "diffusion_offdiag",
                    lower_positions.index(position),
                )
            elif strict_structure:
                raise ValueError(
                    "CORRELATION parameter does not correspond to a modeled latent pair: "
                    f"{parameter.name!r}"
                )

    return (
        offdiag_index,
        lambda_index,
        diag_index,
        diffusion_diag_index,
        diffusion_offdiag_index,
    )


def compile_priors(
    raw_priors: dict[str, dict],
    model_spec: ModelSpec | dict,
    ssm_spec: SSMSpec | None,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    causal_spec: dict | None = None,
) -> tuple[SSMPriors, PriorIndexMaps]:
    """Compile prior proposals into ``SSMPriors`` with explicit index maps."""
    ssm_priors = SSMPriors()

    role_by_name: dict[str, ParameterRole] = {}
    if model_spec:
        if isinstance(model_spec, dict) and model_spec.get("parameters"):
            spec_obj = ModelSpec.model_validate(model_spec)
        elif isinstance(model_spec, ModelSpec):
            spec_obj = model_spec
        else:
            spec_obj = None
        if spec_obj:
            for parameter in spec_obj.parameters:
                role_by_name[parameter.name] = parameter.role

    per_element: dict[str, list[tuple[int, dict[str, float]]]] = {}
    (
        offdiag_param_index,
        lambda_param_index,
        diag_param_index,
        diffusion_diag_param_index,
        diffusion_offdiag_param_index,
    ) = build_prior_index_maps(ssm_spec, model_spec, causal_spec=causal_spec)

    for param_name, prior_spec in raw_priors.items():
        distribution = prior_spec.get("distribution", "Normal")
        params = prior_spec.get("params", {})
        normalized = normalize_prior_params(distribution, params)

        if param_name in diag_param_index:
            attr, idx = diag_param_index[param_name]
            construct_name = param_name.removeprefix("rho_").removeprefix("ar_")
            ref_days = prior_spec.get("reference_interval_days")
            dt = (
                float(ref_days)
                if ref_days is not None and ref_days > 0
                else get_construct_dt_days(
                    causal_spec,
                    construct_name,
                )
            )
            lower = normalized.get("lower")
            upper = normalized.get("upper")
            if lower is not None and float(lower) < 0.0:
                raise ValueError(
                    f"AR prior '{param_name}' must be on the DT persistence scale in [0, 1], "
                    f"but lower bound is {float(lower):.3g}"
                )
            if upper is not None and float(upper) > 1.0:
                raise ValueError(
                    f"AR prior '{param_name}' must be on the DT persistence scale in [0, 1], "
                    f"but upper bound is {float(upper):.3g}"
                )

            mu_ar = float(normalized.get("mu", 0.5))
            if not 0.0 < mu_ar < 1.0:
                raise ValueError(
                    f"AR prior '{param_name}' must have DT persistence mean in (0, 1), got {mu_ar:.3g}"
                )
            mu_ar = min(max(mu_ar, 0.001), 0.999)
            sigma_ar = normalized.get("sigma", 0.2)
            mu_drift = -math.log(mu_ar) / dt
            sigma_drift = sigma_ar / (mu_ar * dt)
            per_element.setdefault(attr, []).append((idx, {"mu": mu_drift, "sigma": sigma_drift}))
            continue

        if param_name in offdiag_param_index:
            attr, idx = offdiag_param_index[param_name]
            ref_days = prior_spec.get("reference_interval_days")
            if ref_days is not None and ref_days > 0:
                dt = float(ref_days)
            else:
                dt = 1.0
                if ssm_spec.latent_names:
                    latent_set = set(ssm_spec.latent_names)
                    compound = param_name.removeprefix("beta_")
                    split = split_compound_name(compound, latent_set, latent_set)
                    if split is not None:
                        _cause, effect = split
                        dt = get_construct_dt_days(causal_spec, effect)
            mu_beta = normalized.get("mu", 0.0)
            sigma_beta = normalized.get("sigma", 0.5)
            per_element.setdefault(attr, []).append(
                (idx, {"mu": mu_beta / dt, "sigma": sigma_beta / dt})
            )
            continue

        if param_name in lambda_param_index:
            attr, idx = lambda_param_index[param_name]
            per_element.setdefault(attr, []).append((idx, normalized))
            continue

        if param_name in diffusion_diag_param_index:
            attr, idx = diffusion_diag_param_index[param_name]
            per_element.setdefault(attr, []).append((idx, normalized))
            continue

        if param_name in diffusion_offdiag_param_index:
            attr, idx = diffusion_offdiag_param_index[param_name]
            per_element.setdefault(attr, []).append((idx, normalized))
            continue

        role = role_by_name.get(param_name)
        if role and role in ROLE_TO_SSM:
            attr, defaults = ROLE_TO_SSM[role]
            merged = {key: normalized.get(key, value) for key, value in defaults.items()}
            setattr(ssm_priors, attr, merged)
            continue

        name_lower = param_name.lower()
        matched = False
        for keywords, attr, defaults in KEYWORD_RULES:
            matching_kw = [kw for kw in keywords if kw in name_lower]
            if not matching_kw:
                continue
            logger.debug(
                "Prior '%s': keyword fallback matched '%s' -> %s",
                param_name,
                matching_kw[0],
                attr,
            )
            merged = {key: normalized.get(key, value) for key, value in defaults.items()}
            setattr(ssm_priors, attr, merged)
            matched = True
            break
        if not matched:
            logger.debug("Prior '%s': no role or keyword match found, skipping", param_name)

    for attr, entries in per_element.items():
        current = getattr(ssm_priors, attr)
        expected_size = expected_prior_size(attr, ssm_spec)
        n_total = max(idx for idx, _ in entries) + 1
        if expected_size is not None:
            n_total = max(n_total, expected_size)

        include_mu = "mu" in current or any("mu" in normalized for _, normalized in entries)
        include_sigma = "sigma" in current or any(
            "sigma" in normalized for _, normalized in entries
        )

        mu_arr = [float(current.get("mu", 0.0))] * n_total if include_mu else None
        sigma_arr = [float(current.get("sigma", 0.5))] * n_total if include_sigma else None

        for idx, normalized in entries:
            if "mu" in normalized and mu_arr is not None:
                mu_arr[idx] = float(normalized["mu"])
            if "sigma" in normalized and sigma_arr is not None:
                sigma_arr[idx] = float(normalized["sigma"])

        result: dict[str, list[float]] = {}
        if mu_arr is not None:
            result["mu"] = mu_arr
        if sigma_arr is not None:
            result["sigma"] = sigma_arr

        has_bounds = any("lower" in normalized for _, normalized in entries)
        if has_bounds:
            lower_arr = [-1e6] * n_total
            upper_arr = [1e6] * n_total
            for idx, normalized in entries:
                lower_arr[idx] = float(normalized.get("lower", -1e6))
                upper_arr[idx] = float(normalized.get("upper", 1e6))
            result["lower"] = lower_arr
            result["upper"] = upper_arr

        setattr(ssm_priors, attr, result)

    warn_first_order_approximation(ssm_priors)
    if ssm_spec is not None:
        check_drift_lag_consistency(ssm_priors, ssm_spec, edge_lag_days=edge_lag_days)

    index_maps = (
        offdiag_param_index,
        lambda_param_index,
        diag_param_index,
        diffusion_diag_param_index,
        diffusion_offdiag_param_index,
    )
    return ssm_priors, index_maps


def bind_parameters(
    model_spec: ModelSpec | dict,
    ssm_spec: SSMSpec,
    index_maps: PriorIndexMaps | None = None,
    *,
    causal_spec: dict | None = None,
) -> list[dict[str, Any]]:
    """Map semantic parameter names to NumPyro sample sites."""
    if index_maps is None:
        (
            offdiag_index,
            lambda_index,
            diag_index,
            diffusion_diag_index,
            diffusion_offdiag_index,
        ) = build_prior_index_maps(ssm_spec, model_spec, causal_spec=causal_spec)
    else:
        (
            offdiag_index,
            lambda_index,
            diag_index,
            diffusion_diag_index,
            diffusion_offdiag_index,
        ) = index_maps

    bindings: list[dict[str, Any]] = []
    ordered_maps = (
        diag_index,
        offdiag_index,
        diffusion_diag_index,
        diffusion_offdiag_index,
        lambda_index,
    )
    for mapping in ordered_maps:
        for param_name, (prior_field, flat_index) in sorted(mapping.items()):
            sample_site = SAMPLE_SITE_FOR_PRIOR_FIELD.get(prior_field)
            if sample_site is None:
                continue
            bindings.append(
                {
                    "parameter": param_name,
                    "site_name": sample_site,
                    "flat_index": flat_index,
                }
            )

    bindings.sort(key=lambda entry: str(entry["parameter"]))
    return bindings


def compile_ssm_inputs(
    model_spec: ModelSpec | dict,
    priors: dict[str, dict],
    causal_spec: dict | None = None,
) -> tuple[SSMSpec, SSMPriors, list[dict[str, Any]]]:
    """Run the full compilation pipeline as a composition of pure stages."""
    spec, edge_lag_days = translate_spec(model_spec, causal_spec)
    ssm_priors, index_maps = compile_priors(
        priors,
        model_spec,
        spec,
        edge_lag_days=edge_lag_days,
        causal_spec=causal_spec,
    )
    bindings = bind_parameters(
        model_spec,
        spec,
        index_maps=index_maps,
        causal_spec=causal_spec,
    )
    return spec, ssm_priors, bindings
