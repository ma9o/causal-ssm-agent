"""Pure spec-translation stage for the SSM compilation pipeline."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.artifacts.statistical_model_spec import (
    DistributionFamily,
    InitializationPolicy,
    LinkFunction,
    ObservationInterceptPolicy,
    ParameterRole,
    StatisticalModelSpec,
)
from nof1_causal_lab.models.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.model_semantics import should_auto_standardize_indicator
from nof1_causal_lab.models.ssm.dynamics.spec import (
    DynamicsSpec,
    HillEdgeSpec,
    LinearEdgeSpec,
    NodePotentialSpec,
    StateInterceptSpec,
)
from nof1_causal_lab.models.ssm.inference.targets.observation_families import (
    supported_distribution_families,
)
from nof1_causal_lab.models.ssm.model import SSMSpec
from nof1_causal_lab.models.ssm.parameter_names import (
    build_initial_state_correlation_support,
    split_compound_name,
)
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from nof1_causal_lab.models.ssm.structure.sites import SiteKind, SupportClass
from nof1_causal_lab.utils.causal_design import (
    build_reference_indicator_lookup,
    get_constructs,
    get_estimation_edges,
    get_estimation_state_order,
    get_indicator_polarity,
    get_indicators,
    get_known_inputs,
    get_marginalized_scales,
)
from nof1_causal_lab.utils.observation_semantics import get_observation_semantics


class SpecTranslationError(AggregatedCompileError):
    """Aggregate independent ``StatisticalModelSpec`` -> ``SSMSpec`` translation errors."""

    header = "Spec translation failed"


def _zero_loading_support(n_manifest: int, n_latent: int) -> np.ndarray:
    return np.zeros((n_manifest, n_latent), dtype=bool)


def _full_vector_support(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def _zero_vector_support(n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool)


def _full_diagonal_support(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def _full_cholesky_support(n: int) -> np.ndarray:
    return np.tri(n, dtype=bool)


def _zero_square_support(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=bool)


def get_construct_dt_days(causal_design: dict | None, _construct_name: str = "") -> float:
    """Get the model clock interval in fractional days."""
    if causal_design is None:
        return 1.0

    model_clock = (
        causal_design.get("measurement", {}).get("model_clock")
        if isinstance(causal_design, dict)
        else getattr(getattr(causal_design, "measurement", None), "model_clock", None)
    )
    if not model_clock:
        return 1.0

    try:
        return parse_duration_to_hours(model_clock) / 24.0
    except ValueError:
        return 1.0


def get_estimation_latent_layout(
    causal_design: dict | None,
) -> tuple[list[str], np.ndarray | None] | None:
    """Build the canonical latent ordering from the retained estimation states."""
    if causal_design is None:
        return None

    try:
        state_order = get_estimation_state_order(causal_design)
    except ValueError as exc:
        raise SpecTranslationError([str(exc)]) from exc
    errors: list[str] = []
    if not state_order:
        errors.append("causal_design.estimation.state_order is empty")
        raise SpecTranslationError(errors)

    latent_construct_lookup = {
        construct["name"]: construct for construct in get_constructs(causal_design)
    }
    unknown_states = [name for name in state_order if name not in latent_construct_lookup]
    if unknown_states:
        errors.append(
            "causal_design.estimation.state_order references constructs absent from latent.constructs: "
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


def get_estimation_input_layout(
    causal_design: dict | None,
) -> tuple[
    list[str],
    list[str],
    list[float],
    list[str],
    list[bool],
]:
    """Build canonical known-input ordering and source metadata."""
    if causal_design is None:
        return [], [], [], [], []
    known_inputs = get_known_inputs(causal_design)
    estimation_edges = get_estimation_edges(causal_design)
    input_lagged: list[bool] = []
    for item in known_inputs:
        name = str(item["construct"])
        edge_lags = {bool(edge["lagged"]) for edge in estimation_edges if edge.get("cause") == name}
        if not edge_lags:
            raise SpecTranslationError(
                [f"Known input {name!r} has no outgoing edge into a retained state"]
            )
        if len(edge_lags) > 1:
            raise SpecTranslationError(
                [
                    f"Known input {name!r} has mixed contemporaneous and lagged outgoing "
                    "edges; one input trajectory must use a consistent alignment"
                ]
            )
        input_lagged.append(edge_lags.pop())
    return (
        [str(item["construct"]) for item in known_inputs],
        [str(item["source_indicator"]) for item in known_inputs],
        [float(item.get("scale", 1.0)) for item in known_inputs],
        [str(item.get("missing_policy", "zero")) for item in known_inputs],
        input_lagged,
    )


def _mask_time_invariant_vector_support(
    mask: np.ndarray,
    time_invariant_mask: np.ndarray | None,
) -> np.ndarray:
    """Drop free vector entries for quasi-static latent states."""
    masked = np.asarray(mask, dtype=bool).copy()
    if time_invariant_mask is None:
        return masked
    masked[np.asarray(time_invariant_mask, dtype=bool)] = False
    return masked


def _mask_time_invariant_drift_targets(
    mask: np.ndarray,
    time_invariant_mask: np.ndarray | None,
) -> np.ndarray:
    """Drop drift off-diagonal entries whose effect state is quasi-static."""
    masked = np.asarray(mask, dtype=bool).copy()
    if time_invariant_mask is None:
        return masked
    masked[np.asarray(time_invariant_mask, dtype=bool), :] = False
    return masked


def _mask_time_invariant_diffusion_support(
    mask: np.ndarray,
    time_invariant_mask: np.ndarray | None,
) -> np.ndarray:
    """Drop diffusion entries that touch quasi-static latent states."""
    masked = np.asarray(mask, dtype=bool).copy()
    if time_invariant_mask is None:
        return masked
    ti = np.asarray(time_invariant_mask, dtype=bool)
    masked[ti, :] = False
    masked[:, ti] = False
    return masked


def build_structural_support_from_causal_design(
    latent_names: list[str] | None,
    manifest_cols: list[str],
    n_latent: int,
    n_manifest: int,
    *,
    manifest_dists: list[DistributionFamily],
    causal_design: dict | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    jnp.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], float],
]:
    """Build block/component support arrays and edge lag metadata from causal structure."""
    if causal_design is None or latent_names is None:
        return (
            np.eye(n_latent, dtype=bool),
            np.zeros((n_latent, 0), dtype=bool),
            jnp.eye(n_manifest, n_latent),
            _zero_loading_support(n_manifest, n_latent),
            np.zeros(n_manifest, dtype=bool),
            {},
        )

    try:
        edges = get_estimation_edges(causal_design)
    except ValueError as exc:
        raise SpecTranslationError([str(exc)]) from exc
    latent_construct_lookup = {
        construct["name"]: construct for construct in get_constructs(causal_design)
    }
    indicators = get_indicators(causal_design)
    errors: list[str] = []

    indicator_names = {
        (indicator.get("name") if isinstance(indicator, dict) else indicator.name)
        for indicator in indicators
    }
    unknown_likelihoods = sorted(set(manifest_cols) - indicator_names)
    if unknown_likelihoods:
        errors.append(
            "StatisticalModelSpec likelihoods reference indicators absent from causal_design measurement: "
            f"{unknown_likelihoods}"
        )

    latent_idx = {name: idx for idx, name in enumerate(latent_names)}
    input_names, _input_sources, _input_scales, _input_policies, _input_lagged = (
        get_estimation_input_layout(causal_design)
    )
    input_idx = {name: idx for idx, name in enumerate(input_names)}
    state_dynamics_support = np.zeros((n_latent, n_latent), dtype=bool)
    input_effect_support = np.zeros((n_latent, len(input_names)), dtype=bool)
    for latent_name, latent_idx_value in latent_idx.items():
        construct = latent_construct_lookup.get(latent_name) or {}
        if construct.get("temporal_status") != "time_invariant":
            state_dynamics_support[latent_idx_value, latent_idx_value] = True
    edge_lag_days: dict[tuple[int, int], float] = {}
    model_dt_days = get_construct_dt_days(causal_design)

    for edge in edges:
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if effect not in latent_idx:
            continue
        if latent_construct_lookup.get(effect, {}).get("temporal_status") == "time_invariant":
            continue
        effect_idx = latent_idx[effect]
        if cause in input_idx:
            input_effect_support[effect_idx, input_idx[cause]] = True
            continue
        if cause not in latent_idx:
            continue
        cause_idx = latent_idx[cause]
        state_dynamics_support[effect_idx, cause_idx] = True

        lagged = edge.get("lagged", True) if isinstance(edge, dict) else edge.lagged
        lag_hours = model_dt_days * 24.0 if lagged else 0.0
        if lag_hours > 0:
            edge_lag_days[(effect_idx, cause_idx)] = lag_hours / 24.0

    manifest_idx = {name: idx for idx, name in enumerate(manifest_cols)}
    lambda_mat_np = np.zeros((n_manifest, n_latent), dtype=np.float64)
    lambda_support = np.zeros((n_manifest, n_latent), dtype=bool)
    reference_indicator_lookup = build_reference_indicator_lookup(indicators)
    matched_manifests: set[str] = set()
    invalid_construct_manifests: set[str] = set()
    construct_channels: dict[str, list[int]] = {}

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
                "CausalDesign measurement indicator references unknown construct: "
                f"{ind_name!r} -> {construct_name!r}"
            )
            invalid_construct_manifests.add(ind_name)
            continue

        manifest_idx_value = manifest_idx[ind_name]
        latent_idx_value = latent_idx[construct_name]
        matched_manifests.add(ind_name)
        construct_channels.setdefault(construct_name, []).append(manifest_idx_value)

        if manifest_dists[manifest_idx_value] == DistributionFamily.CATEGORICAL:
            # Categorical slopes multiply the whole linear predictor, so a free
            # loading is exactly redundant with them (only the products enter
            # the likelihood). The loading is pinned and the slopes carry the
            # channel's discrimination; sign lives in the slopes as well, so
            # polarity is ignored.
            lambda_mat_np[manifest_idx_value, latent_idx_value] = 1.0
        elif ind_name == reference_indicator_lookup.get(construct_name):
            lambda_mat_np[manifest_idx_value, latent_idx_value] = (
                1.0 if get_indicator_polarity(indicator) == "positive" else -1.0
            )
        else:
            lambda_support[manifest_idx_value, latent_idx_value] = True

    lambda_mat = jnp.array(lambda_mat_np)

    # A construct measured only by categorical channels has no fixed-link-scale
    # channel pinning its latent scale, and nominal categories break no
    # reflection symmetry. Pin the reference channel's first non-baseline slope
    # to +1 as the construct's scale and sign anchor (Bock-NRM style).
    manifest_cat_anchor = np.zeros(n_manifest, dtype=bool)
    for construct_name, channel_indices in construct_channels.items():
        if not all(
            manifest_dists[channel] == DistributionFamily.CATEGORICAL for channel in channel_indices
        ):
            continue
        reference_name = reference_indicator_lookup.get(construct_name)
        reference_channel = manifest_idx.get(reference_name or "")
        if reference_channel in channel_indices:
            manifest_cat_anchor[reference_channel] = True

    unmatched_manifests = sorted(
        set(manifest_cols)
        - matched_manifests
        - set(unknown_likelihoods)
        - invalid_construct_manifests
    )
    if unmatched_manifests:
        errors.append(
            "StatisticalModelSpec likelihoods could not be mapped to causal_design measurement indicators: "
            f"{unmatched_manifests}"
        )

    if errors:
        raise SpecTranslationError(errors)

    return (
        state_dynamics_support,
        input_effect_support,
        lambda_mat,
        lambda_support,
        manifest_cat_anchor,
        edge_lag_days,
    )


def build_manifest_variance_from_causal_design(
    latent_names: list[str] | None,
    manifest_cols: list[str],
    manifest_dists: list[DistributionFamily],
    *,
    causal_design: dict | None,
) -> tuple[jnp.ndarray, np.ndarray]:
    """Build manifest-noise structure from the retained measurement structure.

    A diagonal manifest-noise entry is free only when both conditions hold:
    - the construct has more than one indicator (single-indicator constructs
      absorb measurement error into the structural residual, so their manifest
      channels get fixed zero observation noise), and
    - the indicator's observation family actually reads per-channel noise in
      its emission log-prob (see ``DistributionFamily.uses_manifest_noise``).
      Non-Gaussian, non-Student-t families (Poisson, Gamma, Bernoulli,
      Negative-Binomial, Beta, Ordered-Logistic, Categorical) ignore R, so a
      free noise site would be a disconnected parameter.
    """
    n_manifest = len(manifest_cols)
    empty_variance = jnp.zeros((n_manifest, n_manifest))
    family_noise_mask = np.array(
        [dist.uses_manifest_noise for dist in manifest_dists],
        dtype=bool,
    )

    if causal_design is None or latent_names is None:
        return empty_variance, family_noise_mask

    indicators = get_indicators(causal_design)
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
        return empty_variance, family_noise_mask

    manifest_var_mask = np.zeros(n_manifest, dtype=bool)
    for manifest_name, construct_name in manifest_to_construct.items():
        if indicators_per_construct.get(construct_name, 0) <= 1:
            continue
        idx = manifest_idx[manifest_name]
        if not manifest_dists[idx].uses_manifest_noise:
            continue
        manifest_var_mask[idx] = True

    manifest_var = np.zeros((n_manifest, n_manifest), dtype=np.float64)
    return jnp.array(manifest_var), manifest_var_mask


def build_manifest_level_counts_from_causal_design(
    manifest_cols: list[str],
    manifest_dists: list[DistributionFamily],
    *,
    causal_design: dict | None,
) -> list[int] | None:
    """Build per-manifest discrete level counts from causal-design metadata."""
    if causal_design is None:
        return None

    needs_level_metadata = any(
        dist in {DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL}
        for dist in manifest_dists
    )
    if not needs_level_metadata:
        return None

    indicator_lookup = {
        (indicator.get("name") if isinstance(indicator, dict) else indicator.name): indicator
        for indicator in get_indicators(causal_design)
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
                f"Indicator '{manifest_name}' uses ordered_logistic but causal_design is missing "
                "ordinal_levels with at least 2 levels"
            )
            continue
        level_counts[idx] = len(ordinal_levels)

    if errors:
        raise SpecTranslationError(errors)
    return level_counts


def _build_role_index_lookup(
    statistical_model_spec: StatisticalModelSpec,
    *,
    role: ParameterRole,
    prefix: str,
    names: list[str],
) -> np.ndarray:
    """Build a vector mask from active semantic parameters sharing a name prefix."""
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    mask = np.zeros(len(names), dtype=bool)
    for parameter in statistical_model_spec.parameters:
        if parameter.role != role:
            continue
        resolved_name = parameter.name.removeprefix(prefix)
        idx = name_to_idx.get(resolved_name)
        if idx is not None:
            mask[idx] = True
    return mask


def _hill_edge_targets(
    statistical_model_spec: StatisticalModelSpec,
    latent_names: list[str],
) -> set[tuple[int, int]]:
    """Structural edges the author gave a saturating (Hill) form.

    An edge is Hill when the StatisticalModelSpec carries a ``hill_emax_<cause>_<effect>``
    parameter; its EC50 and Hill-coefficient parameters share the same
    ``hill_ec50_``/``hill_n_`` naming. Returns ``(cause_idx, effect_idx)`` pairs.
    """
    latent_name_set = set(latent_names)
    name_to_idx = {name: idx for idx, name in enumerate(latent_names)}
    targets: set[tuple[int, int]] = set()
    for parameter in statistical_model_spec.parameters:
        if not parameter.name.startswith("hill_emax_"):
            continue
        parsed = split_compound_name(
            parameter.name.removeprefix("hill_emax_"),
            latent_name_set,
            latent_name_set,
        )
        if parsed is None:
            continue
        cause_name, effect_name = parsed
        targets.add((name_to_idx[cause_name], name_to_idx[effect_name]))
    return targets


def _build_manifest_standardized_flags(
    statistical_model_spec: StatisticalModelSpec,
    manifest_cols: list[str],
    *,
    causal_design: dict | None,
) -> list[bool]:
    """Return deterministic standardization tags for each manifest channel."""
    likelihood_lookup = {
        likelihood.variable: likelihood for likelihood in statistical_model_spec.likelihoods
    }
    indicator_lookup = {}
    if causal_design is not None:
        indicator_lookup = {
            indicator["name"]: indicator for indicator in get_indicators(causal_design)
        }

    standardized: list[bool] = []
    for manifest_name in manifest_cols:
        likelihood = likelihood_lookup[manifest_name]
        indicator = indicator_lookup.get(manifest_name) or {}
        support_kind = indicator.get("support_kind")
        summary_operator = indicator.get("summary_operator")
        if indicator and (
            not isinstance(support_kind, str) or not isinstance(summary_operator, str)
        ):
            semantics = get_observation_semantics(indicator)
            support_kind = semantics.support_kind.value
            summary_operator = semantics.summary_operator.value

        if isinstance(support_kind, str) and isinstance(summary_operator, str):
            standardized.append(
                should_auto_standardize_indicator(
                    likelihood.distribution,
                    likelihood.link,
                    support_kind,
                    summary_operator,
                )
            )
            continue

        standardized.append(bool(likelihood.standardized))
    return standardized


def _latent_standardized_anchor_mask(
    latent_names: list[str],
    manifest_cols: list[str],
    manifest_standardized: list[bool],
    *,
    causal_design: dict | None,
) -> np.ndarray:
    """Mark latents whose location is pinned by a standardized manifest channel."""
    mask = np.zeros(len(latent_names), dtype=bool)
    if causal_design is None:
        return mask

    latent_idx = {name: idx for idx, name in enumerate(latent_names)}
    standardized_lookup = dict(zip(manifest_cols, manifest_standardized, strict=True))
    for indicator in get_indicators(causal_design):
        ind_name = indicator.get("name") if isinstance(indicator, dict) else indicator.name
        construct_name = (
            indicator.get("construct_name")
            if isinstance(indicator, dict)
            else indicator.construct_name
        )
        latent_index = latent_idx.get(construct_name)
        if latent_index is not None and standardized_lookup.get(ind_name):
            mask[latent_index] = True
    return mask


def _build_static_factor_structure(
    statistical_model_spec: StatisticalModelSpec,
    latent_names: list[str],
    *,
    causal_design: dict | None,
) -> tuple[np.ndarray, jnp.ndarray, jnp.ndarray, list[str]]:
    """Compile deterministic baseline-factor loadings from marginalized scales."""
    factor_names = [
        parameter.name
        for parameter in statistical_model_spec.parameters
        if parameter.role == ParameterRole.STATIC_STATE_SD
    ]
    if not factor_names:
        return (
            np.zeros(0, dtype=bool),
            jnp.zeros(0),
            jnp.zeros((len(latent_names), 0)),
            [],
        )

    if causal_design is None:
        raise SpecTranslationError(
            [
                "STATIC_STATE_SD parameters require causal_design so baseline factors can be "
                "compiled from induced time-invariant confounders."
            ]
        )

    scales_by_name = {
        scale["parameter"]: scale
        for scale in get_marginalized_scales(causal_design)
        if scale["kind"] == "initial_state_correlation"
    }

    latent_idx = {name: idx for idx, name in enumerate(latent_names)}
    loadings = np.zeros((len(latent_names), len(factor_names)), dtype=np.float64)
    errors: list[str] = []

    for factor_idx, factor_name in enumerate(factor_names):
        scale = scales_by_name.get(factor_name)
        if scale is None:
            errors.append(
                "STATIC_STATE_SD parameter does not match any marginalized "
                f"initial-state-correlation scale: {factor_name!r}"
            )
            continue
        for state_name in scale["affected_states"]:
            latent_idx_value = latent_idx.get(state_name)
            if latent_idx_value is not None:
                loadings[latent_idx_value, factor_idx] = 1.0

    if errors:
        raise SpecTranslationError(errors)

    return (
        np.ones(len(factor_names), dtype=bool),
        jnp.zeros(len(factor_names)),
        jnp.asarray(loadings),
        factor_names,
    )


def translate_spec(
    statistical_model_spec: StatisticalModelSpec | dict,
    causal_design: dict | None = None,
) -> tuple[SSMSpec, dict[tuple[int, int], float]]:
    """Translate ``StatisticalModelSpec`` into ``SSMSpec`` with explicit edge-lag metadata.

    Assumes the caller has already validated ``statistical_model_spec``. This function
    is a pure translation stage — it does not re-validate.
    """
    if isinstance(statistical_model_spec, dict):
        statistical_model_spec = StatisticalModelSpec.model_validate(statistical_model_spec)

    manifest_cols = [lik.variable for lik in statistical_model_spec.likelihoods]
    n_manifest = len(manifest_cols)
    errors: list[str] = []

    layout_failed = False
    try:
        structural_layout = get_estimation_latent_layout(causal_design)
    except SpecTranslationError as exc:
        structural_layout = None
        layout_failed = True
        errors.extend(exc.errors)
    if structural_layout is not None:
        latent_names, time_invariant_mask = structural_layout
        n_latent = len(latent_names)
    else:
        if causal_design is not None and layout_failed:
            latent_names = []
            time_invariant_mask = None
            n_latent = 0
        else:
            ar_params = [
                param
                for param in statistical_model_spec.parameters
                if param.role == ParameterRole.AR_COEFFICIENT
            ]
            if not ar_params:
                errors.append(
                    "No AR_COEFFICIENT parameters found in StatisticalModelSpec; "
                    "cannot infer latent dimensionality without causal_design."
                )
                raise SpecTranslationError(errors)
            n_latent = len(ar_params)
            latent_names = [param.name.removeprefix("rho_") for param in ar_params]
            time_invariant_mask = None

    manifest_dists: list[DistributionFamily] = []
    supported_families = supported_distribution_families()
    for likelihood in statistical_model_spec.likelihoods:
        dist = likelihood.distribution
        if dist not in supported_families:
            supported = sorted(distribution.value for distribution in supported_families)
            errors.append(
                f"Indicator '{likelihood.variable}': distribution '{dist}' "
                f"has no native emission function. Supported: {supported}."
            )
        manifest_dists.append(dist)

    if causal_design is not None and layout_failed:
        raise SpecTranslationError(errors)

    manifest_links: list[LinkFunction] = [
        likelihood.link for likelihood in statistical_model_spec.likelihoods
    ]

    try:
        (
            state_dynamics_support,
            input_effect_support,
            lambda_mat,
            lambda_support,
            manifest_cat_anchor,
            edge_lag_days,
        ) = build_structural_support_from_causal_design(
            latent_names,
            manifest_cols,
            n_latent,
            n_manifest,
            manifest_dists=manifest_dists,
            causal_design=causal_design,
        )
    except SpecTranslationError as exc:
        errors.extend(exc.errors)
        state_dynamics_support = np.eye(n_latent, dtype=bool)
        input_effect_support = np.zeros((n_latent, 0), dtype=bool)
        lambda_mat = jnp.eye(n_manifest, n_latent)
        lambda_support = _zero_loading_support(n_manifest, n_latent)
        manifest_cat_anchor = np.zeros(n_manifest, dtype=bool)
        edge_lag_days = {}

    if causal_design is None:
        latent_name_set = set(latent_names)
        latent_idx = {name: idx for idx, name in enumerate(latent_names)}
        for parameter in statistical_model_spec.parameters:
            if parameter.role != ParameterRole.FIXED_EFFECT:
                continue
            parsed = split_compound_name(
                parameter.name.removeprefix("beta_"),
                latent_name_set,
                latent_name_set,
            )
            if parsed is None:
                continue
            cause_name, effect_name = parsed
            state_dynamics_support[latent_idx[effect_name], latent_idx[cause_name]] = True

    decay_support = np.diag(state_dynamics_support).copy()
    decay_support = _mask_time_invariant_vector_support(decay_support, time_invariant_mask)
    linear_edge_support = np.asarray(state_dynamics_support, dtype=bool).copy()
    np.fill_diagonal(linear_edge_support, False)
    linear_edge_support = _mask_time_invariant_drift_targets(
        linear_edge_support,
        time_invariant_mask,
    )

    manifest_chol, manifest_chol_diag_support = build_manifest_variance_from_causal_design(
        latent_names,
        manifest_cols,
        manifest_dists,
        causal_design=causal_design,
    )
    try:
        manifest_level_counts = build_manifest_level_counts_from_causal_design(
            manifest_cols,
            manifest_dists,
            causal_design=causal_design,
        )
    except SpecTranslationError as exc:
        errors.extend(exc.errors)
        manifest_level_counts = None

    has_innovation_correlation = any(
        parameter.role == ParameterRole.CORRELATION
        for parameter in statistical_model_spec.parameters
    )
    if causal_design is not None and any(
        parameter.role == ParameterRole.INITIAL_STATE_CORRELATION
        for parameter in statistical_model_spec.parameters
    ):
        errors.append(
            "Causal-spec compilation no longer accepts INITIAL_STATE_CORRELATION parameters; "
            "use compiled STATIC_STATE_SD baseline factors instead."
        )
    try:
        if causal_design is None:
            t0_correlation_support = build_initial_state_correlation_support(
                latent_names, statistical_model_spec
            )
        else:
            t0_correlation_support = _zero_square_support(n_latent)
    except ValueError as exc:
        errors.append(str(exc))
        t0_correlation_support = _zero_square_support(n_latent)
    diffusion_chol_support = (
        _full_cholesky_support(n_latent)
        if has_innovation_correlation
        else np.diag(_full_diagonal_support(n_latent))
    )
    diffusion_chol_support = _mask_time_invariant_diffusion_support(
        diffusion_chol_support,
        time_invariant_mask,
    )
    if t0_correlation_support is None:
        t0_correlation_support = _zero_square_support(n_latent)
    initialization_policy = InitializationPolicy(statistical_model_spec.initialization_policy)
    if initialization_policy == InitializationPolicy.FREE:
        t0_means_support = _full_vector_support(n_latent)
        t0_chol_diag_support = _full_diagonal_support(n_latent)
    else:
        dynamic_mask = (
            np.ones(n_latent, dtype=bool)
            if time_invariant_mask is None
            else ~np.asarray(time_invariant_mask, dtype=bool)
        )
        t0_means_support = np.zeros(n_latent, dtype=bool)
        t0_means_support[~dynamic_mask] = True
        t0_chol_diag_support = np.zeros(n_latent, dtype=bool)
        t0_chol_diag_support[~dynamic_mask] = True

    observation_intercept_policy = ObservationInterceptPolicy(
        statistical_model_spec.observation_intercept_policy
    )
    if observation_intercept_policy == ObservationInterceptPolicy.FIXED:
        manifest_means_support = _zero_vector_support(n_manifest)
    else:
        manifest_means_support = _build_role_index_lookup(
            statistical_model_spec,
            role=ParameterRole.OBSERVATION_INTERCEPT,
            prefix="manifest_mean_",
            names=manifest_cols,
        )
    if statistical_model_spec.equilibrium_forcing:
        state_intercept_support = _build_role_index_lookup(
            statistical_model_spec,
            role=ParameterRole.STATE_INTERCEPT,
            prefix="cint_",
            names=latent_names,
        )
    else:
        state_intercept_support = _zero_vector_support(n_latent)
    static_state_sd_support, static_state_sds, static_factor_loadings, static_factor_names = (
        _build_static_factor_structure(
            statistical_model_spec,
            latent_names,
            causal_design=causal_design,
        )
    )
    input_names, input_sources, input_scales, input_policies, input_lagged = (
        get_estimation_input_layout(causal_design)
    )
    manifest_standardized = _build_manifest_standardized_flags(
        statistical_model_spec,
        manifest_cols,
        causal_design=causal_design,
    )

    # Time-invariant constructs have no dynamics anchor (no potential well),
    # so a free t0 mean rides an exact additive ridge with the channel-side
    # location parameters unless a standardized channel pins the construct's
    # location (see docs/reference/statistical-model-spec/identification.md).
    latent_standardized_anchor = _latent_standardized_anchor_mask(
        latent_names,
        manifest_cols,
        manifest_standardized,
        causal_design=causal_design,
    )
    if time_invariant_mask is not None:
        static_mask = np.asarray(time_invariant_mask, dtype=bool)
        t0_means_support[static_mask & ~latent_standardized_anchor] = False

    if errors:
        raise SpecTranslationError(errors)

    # Self-dynamics lives on the node, not the adjacency diagonal: each
    # self-regulated latent becomes a quadratic potential well (NodePotential),
    # folding the former StateDecay (stiffness = relaxation rate) and the
    # set-point role of StateIntercept (center = well minimum). The center is
    # free only when a state intercept was requested (equilibrium forcing);
    # otherwise the well is pinned at 0 (relaxation toward 0, as before). The
    # cubic self-limitation (quartic) is freed only for constructs the author
    # flagged self-limiting (a ``self_limit_<latent>`` parameter); otherwise it
    # stays pinned at 0 (pure linear relaxation).
    self_limit_support = _build_role_index_lookup(
        statistical_model_spec,
        role=ParameterRole.DYNAMICS_PARAMETER_POSITIVE,
        prefix="self_limit_",
        names=latent_names,
    )
    hill_edge_targets = _hill_edge_targets(statistical_model_spec, latent_names)
    dynamics_components = []
    for latent_idx in range(n_latent):
        has_well = bool(decay_support[latent_idx])
        has_setpoint = bool(state_intercept_support[latent_idx])
        if has_well:
            dynamics_components.append(
                NodePotentialSpec(
                    target=latent_idx,
                    fixed_center=None if has_setpoint else 0.0,
                    fixed_stiffness=None,
                    fixed_quartic=None if bool(self_limit_support[latent_idx]) else 0.0,
                )
            )
        elif has_setpoint:
            # Intercept without relaxation is a constant forcing term (a ramp),
            # not a set-point — keep it as an explicit StateIntercept.
            dynamics_components.append(StateInterceptSpec(target=latent_idx))
    # Each structural edge is materialized as a linear weight unless the author
    # flagged it saturating (Hill parameters present), in which case it becomes a
    # Hill dose-response component. Exactly one component per structural edge.
    for effect_idx, cause_idx in zip(*np.where(linear_edge_support), strict=False):
        source, target = int(cause_idx), int(effect_idx)
        if (source, target) in hill_edge_targets:
            dynamics_components.append(HillEdgeSpec(source=source, target=target))
        else:
            dynamics_components.append(LinearEdgeSpec(source=source, target=target))

    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=DynamicsSpec(
            n_latent=n_latent,
            components=tuple(dynamics_components),
        ),
        diffusion_block=DiffusionBlockSpec(
            n_latent=n_latent,
            diffusion_chol_support=diffusion_chol_support,
            diffusion_chol_template=jnp.eye(n_latent),
            time_invariant_mask=time_invariant_mask,
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            free_support=lambda_support,
            template=lambda_mat,
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=n_manifest,
            free_support=manifest_means_support,
            template=jnp.zeros(n_manifest),
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=n_manifest,
            diag_support=manifest_chol_diag_support,
            template=manifest_chol,
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            free_support=t0_means_support,
            template=jnp.zeros(n_latent),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=n_latent,
            diag_support=t0_chol_diag_support,
            correlation_support=t0_correlation_support,
            template=jnp.eye(n_latent),
        ),
        input_effect_block=SparseMatrixBlockSpec(
            n_rows=n_latent,
            n_cols=len(input_names),
            free_support=input_effect_support,
            template=jnp.zeros((n_latent, len(input_names))),
            free_site_name="input_effect_free",
            det_site_name="input_effect",
            support=SupportClass.REAL,
            site_kind=SiteKind.INPUT_EFFECT,
            assembly_group="input_effect",
            fixed_spec_field="input_effect",
            priors_field="input_effect",
        ),
        static_state_sd_block=SparseVectorBlockSpec(
            n=int(jnp.asarray(static_factor_loadings).shape[1]),
            free_support=static_state_sd_support,
            template=static_state_sds,
            free_site_name="static_state_sd_free",
            det_site_name="static_state_sds",
            support=SupportClass.POSITIVE,
            site_kind=SiteKind.STATIC_STATE_SD,
            assembly_group="t0",
            fixed_spec_field="static_state_sds",
            priors_field="static_state_sd",
        ),
        static_factor_loadings=static_factor_loadings,
        diffusion_dists=[DistributionFamily.GAUSSIAN] * n_latent,
        manifest_dists=manifest_dists,
        manifest_links=manifest_links,
        manifest_standardized=manifest_standardized,
        manifest_cat_anchor=[bool(flag) for flag in manifest_cat_anchor],
        manifest_level_counts=manifest_level_counts,
        latent_names=latent_names,
        manifest_names=manifest_cols,
        input_names=input_names,
        input_source_indicators=input_sources,
        input_scales=input_scales,
        input_missing_policies=input_policies,
        input_lagged=input_lagged,
        static_factor_names=static_factor_names,
        initialization_policy=initialization_policy.value,
        observation_intercept_policy=observation_intercept_policy.value,
    )
    return spec, edge_lag_days
