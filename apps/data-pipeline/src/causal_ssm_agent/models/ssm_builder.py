"""SSM Model Builder for causal SSM pipeline integration.

Provides a model builder interface compatible with the causal SSM pipeline
while using the NumPyro SSM implementation underneath.
"""

import math
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import polars as pl

from causal_ssm_agent.flows import get_prefect_logger
from causal_ssm_agent.models.likelihoods.base import NUMERICAL_EPSILON
from causal_ssm_agent.models.ssm import (
    InferenceResult,
    SSMModel,
    SSMPriors,
    SSMSpec,
    fit,
)
from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionFamily,
    LinkFunction,
    ModelSpec,
    ParameterRole,
)
from causal_ssm_agent.utils.causal_spec import get_constructs, get_edges, get_indicators
from causal_ssm_agent.workers.schemas_prior import PriorProposal

logger = get_prefect_logger(__name__)

# Distributions that have native emission functions in emissions.py.
_SUPPORTED_EMISSIONS: set[DistributionFamily] = {
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


# Map ParameterRole to SSMPriors field and default mu/sigma params.
# This replaces the old keyword-matching _PRIOR_RULES.
_ROLE_TO_SSM: dict[ParameterRole, tuple[str, dict]] = {
    ParameterRole.AR_COEFFICIENT: ("drift_diag", {"mu": -0.5, "sigma": 1.0}),
    ParameterRole.FIXED_EFFECT: ("drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    ParameterRole.RESIDUAL_SD: ("diffusion_diag", {"sigma": 1.0}),
    ParameterRole.LOADING: ("lambda_free", {"mu": 0.5, "sigma": 0.5}),
    ParameterRole.CORRELATION: ("diffusion_offdiag", {"mu": 0.0, "sigma": 0.5}),
}

# Fallback keyword matching for parameters without a role in the ModelSpec
# (e.g. when priors are provided as a flat dict without ParameterSpec context)
_KEYWORD_RULES: list[tuple[list[str], str, dict]] = [
    (["rho", "ar"], "drift_diag", {"mu": -0.5, "sigma": 1.0}),
    (["beta"], "drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    (["sigma", "sd"], "diffusion_diag", {"sigma": 1.0}),
    (["lambda", "loading"], "lambda_free", {"mu": 0.5, "sigma": 0.5}),
    (["cor"], "diffusion_offdiag", {"mu": 0.0, "sigma": 0.5}),
]

_SAMPLE_SITE_FOR_PRIOR_FIELD: dict[str, str] = {
    "drift_diag": "drift_diag_pop",
    "drift_offdiag": "drift_offdiag_pop",
    "diffusion_diag": "diffusion_diag_pop",
    "diffusion_offdiag": "diffusion_lower",
    "lambda_free": "lambda_free",
}

_NON_MANIFEST_COLUMNS = {"time", "time_bucket"}


def _normalize_prior_params(distribution: str, params: dict) -> dict:
    """Convert distribution-specific params to mu/sigma for SSMPriors.

    SSMPriors always uses mu/sigma dicts. This converts from other
    distribution parameterizations (Beta alpha/beta, Uniform lower/upper, etc.).

    Args:
        distribution: Distribution name (Normal, Beta, HalfNormal, etc.)
        params: Original distribution parameters

    Returns:
        Dict with mu and/or sigma keys
    """
    dist_lower = distribution.lower()

    if dist_lower == "normal" or dist_lower == "truncatednormal":
        return {"mu": params.get("mu", 0.0), "sigma": params.get("sigma", 1.0)}

    if dist_lower == "halfnormal":
        return {"sigma": params.get("sigma", 1.0)}

    if dist_lower == "beta":
        alpha = params.get("alpha", 2.0)
        beta = params.get("beta", 2.0)
        # Convert Beta(alpha, beta) to approximate Normal(mu, sigma)
        # E[X] = alpha / (alpha + beta), Var[X] = alpha*beta / ((a+b)^2*(a+b+1))
        mu = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return {"mu": mu, "sigma": var**0.5}

    if dist_lower == "uniform":
        lower = params.get("lower", -1.0)
        upper = params.get("upper", 1.0)
        # Convert Uniform to TruncatedNormal to preserve hard bounds
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 4
        return {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper}

    # Fallback: try to extract mu/sigma directly
    return {"mu": params.get("mu", 0.0), "sigma": params.get("sigma", 1.0)}


def _split_compound_name(
    compound: str,
    valid_first: set[str],
    valid_second: set[str],
) -> tuple[str, str] | None:
    """Split a compound name into two known names.

    Tries all possible split positions and returns the first pair where both
    parts are in the valid sets.  Handles multi-word construct names like
    ``stress_level_focus_quality`` → ``("stress_level", "focus_quality")``.

    Args:
        compound: The underscore-joined string (prefix already removed).
        valid_first: Valid names for the first part.
        valid_second: Valid names for the second part.

    Returns:
        ``(first, second)`` or ``None`` if no valid split exists.
    """
    parts = compound.split("_")
    for i in range(1, len(parts)):
        first = "_".join(parts[:i])
        second = "_".join(parts[i:])
        if first in valid_first and second in valid_second:
            return first, second
    return None


def _default_manifest_columns(X: Any) -> list[str]:
    """Infer manifest columns from a wide dataframe-like object."""
    return [c for c in X.columns if c not in _NON_MANIFEST_COLUMNS and not str(c).endswith("_lag1")]


def _resolve_manifest_metadata(spec: SSMSpec, X: Any) -> tuple[list[str], list[DistributionFamily]]:
    """Resolve manifest column names and distribution families from spec and data."""
    manifest_dists = spec.manifest_dists or [spec.manifest_dist] * spec.n_manifest
    manifest_cols = spec.manifest_names or _default_manifest_columns(X)
    if len(manifest_cols) != spec.n_manifest:
        raise ValueError(
            "Wide data columns do not match SSMSpec manifest dimensionality: "
            f"{len(manifest_cols)} vs {spec.n_manifest}"
        )
    return manifest_cols, manifest_dists


def _extract_numeric_column_values(X: Any, column: str) -> np.ndarray:
    """Extract one manifest column as float64, dropping nulls but not infinities."""
    if isinstance(X, pl.DataFrame):
        values = X.select(pl.col(column).cast(pl.Float64, strict=False)).to_series().to_numpy()
    else:
        series = X[column]
        if hasattr(series, "to_numpy"):
            try:
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
            except TypeError:
                values = series.to_numpy()
        else:
            values = np.asarray(series)
        values = np.asarray(values, dtype=np.float64)

    return values[~np.isnan(values)]


class SSMModelBuilder:
    """Model builder for SSM using NumPyro.

    This class provides an interface compatible with the causal SSM pipeline,
    translating from the ModelSpec to SSMSpec internally.
    """

    _model_type = "SSM"
    version = "0.1.0"

    def __init__(
        self,
        model_spec: ModelSpec | dict | None = None,
        priors: dict[str, PriorProposal] | dict[str, dict] | None = None,
        ssm_spec: SSMSpec | None = None,
        ssm_priors: SSMPriors | None = None,
        compiled_prior_semantics: dict | None = None,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        causal_spec: dict | None = None,
        parameter_bindings: list[dict[str, Any]] | None = None,
    ):
        """Initialize the SSM model builder.

        Args:
            model_spec: Model specification from orchestrator (will be converted)
            priors: Prior proposals for each parameter
            ssm_spec: Direct SSMSpec (overrides model_spec conversion)
            ssm_priors: Direct SSMPriors paired with ssm_spec
            model_config: Override model configuration (n_particles, pf_seed)
            sampler_config: Override sampler configuration
            causal_spec: CausalSpec dict with latent model edges and measurement
                model indicators. When provided, _convert_spec_to_ssm builds
                drift_mask and lambda_mask from the DAG structure.
        """
        self._model_spec = model_spec
        self._priors = priors or {}
        self._ssm_spec = ssm_spec
        self._ssm_priors = ssm_priors
        self._compiled_prior_semantics = compiled_prior_semantics
        self._model_config = model_config or {}
        self._sampler_config = sampler_config or self.get_default_sampler_config()
        self._causal_spec = causal_spec
        self._parameter_bindings = parameter_bindings

        self._model: SSMModel | None = None
        self._spec: SSMSpec | None = None
        self._result: InferenceResult | None = None
        self._edge_lag_days: dict[tuple[int, int], float] = {}

    def compile_inputs(self) -> tuple[SSMSpec, SSMPriors]:
        """Compile user-facing specs into executable SSM inputs."""
        edge_lag_days: dict[tuple[int, int], float] = {}
        index_maps = None

        if self._ssm_spec is not None:
            spec = self._ssm_spec
        elif self._model_spec is not None:
            spec, edge_lag_days = self._convert_spec_to_ssm(self._model_spec)
        else:
            raise ValueError("Cannot compile SSM inputs without model_spec or ssm_spec")

        if self._ssm_priors is not None:
            priors = self._ssm_priors
        else:
            raw_priors: dict[str, dict] = {
                k: v.model_dump() if isinstance(v, PriorProposal) else v
                for k, v in self._priors.items()
            }
            priors, index_maps = self._convert_priors_to_ssm(
                raw_priors, self._model_spec or {}, ssm_spec=spec, edge_lag_days=edge_lag_days
            )

        if self._parameter_bindings is None and self._model_spec is not None:
            self._parameter_bindings = self._compile_parameter_bindings(
                spec, self._model_spec, index_maps=index_maps
            )

        return spec, priors

    @staticmethod
    def get_default_sampler_config() -> dict:
        """Default sampler configuration, read from config.yaml."""
        from causal_ssm_agent.utils.config import get_config

        return get_config().inference.to_sampler_config()

    def _get_construct_dt_days(self, _construct_name: str = "") -> float:
        """Get the time-step size in fractional days.

        Uses ``model_clock`` from the measurement model.
        Falls back to 1.0 (daily) when no spec is available.
        """
        if self._causal_spec is None:
            return 1.0
        model_clock = (
            self._causal_spec.get("measurement", {}).get("model_clock")
            if isinstance(self._causal_spec, dict)
            else getattr(getattr(self._causal_spec, "measurement", None), "model_clock", None)
        )
        if model_clock:
            try:
                from causal_ssm_agent.orchestrator.schemas import parse_duration_to_hours

                return parse_duration_to_hours(model_clock) / 24.0
            except ValueError:
                return 1.0
        return 1.0

    def _get_structural_latent_layout(self) -> tuple[list[str], np.ndarray | None] | None:
        """Build canonical latent ordering from CausalSpec when available.

        The causal structure owns latent identity. Time-varying constructs define
        the dynamic state, and time-invariant constructs are appended as
        quasi-constant latents.
        """
        if self._causal_spec is None:
            return None

        constructs = get_constructs(self._causal_spec)
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

    @staticmethod
    def _expected_prior_size(attr: str, ssm_spec: SSMSpec | None) -> int | None:
        """Return the structural size for an array-valued prior field."""
        if ssm_spec is None:
            return None

        if attr == "drift_diag":
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
            n = ssm_spec.n_latent
            return n * (n - 1) // 2

        return None

    def _convert_spec_to_ssm(
        self, model_spec: ModelSpec | dict
    ) -> tuple[SSMSpec, dict[tuple[int, int], float]]:
        """Convert ModelSpec to SSMSpec.

        When causal_spec is provided, builds drift_mask and lambda_mask from
        the DAG structure instead of using a fully free drift and identity lambda.

        Args:
            model_spec: Model specification

        Returns:
            (SSMSpec, edge_lag_days) — the spec for continuous-time model and
            a dict mapping (effect_idx, cause_idx) → lag in days.
        """
        if isinstance(model_spec, dict):
            from causal_ssm_agent.orchestrator.schemas_model import ModelSpec

            model_spec = ModelSpec.model_validate(model_spec)

        from causal_ssm_agent.models.ssm_compiler import validate_model_spec_for_compilation

        validated_model_spec, errors = validate_model_spec_for_compilation(
            model_spec, causal_spec=self._causal_spec
        )
        if errors:
            raise ValueError("ModelSpec failed compiler validation:\n" + "\n".join(errors))
        assert validated_model_spec is not None
        model_spec = validated_model_spec

        # Extract dimensions from data
        manifest_cols = [lik.variable for lik in model_spec.likelihoods]
        n_manifest = len(manifest_cols)

        structural_layout = self._get_structural_latent_layout()
        if structural_layout is not None:
            latent_names, time_invariant_mask = structural_layout
            n_latent = len(latent_names)
        else:
            # Infer latent structure from parameters only when no causal structure exists.
            ar_params = [p for p in model_spec.parameters if p.role == ParameterRole.AR_COEFFICIENT]
            if not ar_params:
                raise ValueError(
                    "No AR_COEFFICIENT parameters found in ModelSpec; "
                    "cannot infer latent dimensionality without causal_spec."
                )
            n_latent = len(ar_params)
            latent_names = [p.name.removeprefix("rho_") for p in ar_params] if ar_params else None
            time_invariant_mask = None

        # Determine per-indicator noise families from likelihoods.
        # Distributions are passed through directly — no approximation.
        manifest_dists: list[DistributionFamily] = []
        for lik in model_spec.likelihoods:
            dist = lik.distribution
            if dist not in _SUPPORTED_EMISSIONS:
                raise ValueError(
                    f"Indicator '{lik.variable}': distribution '{dist}' "
                    f"has no native emission function. Supported: "
                    f"{sorted(d.value for d in _SUPPORTED_EMISSIONS)}."
                )
            manifest_dists.append(dist)

        # Scalar fallback: first non-Gaussian type (for PF dispatch)
        manifest_dist = DistributionFamily.GAUSSIAN
        for nd in manifest_dists:
            if nd != DistributionFamily.GAUSSIAN:
                manifest_dist = nd
                break

        # Extract per-channel link functions from likelihoods
        manifest_links: list[LinkFunction] = [lik.link for lik in model_spec.likelihoods]

        # Scalar fallback: first non-identity link
        manifest_link = LinkFunction.IDENTITY
        for lk in manifest_links:
            if lk != LinkFunction.IDENTITY:
                manifest_link = lk
                break

        # Build masks from causal_spec if available
        drift_mask, lambda_mat, lambda_mask, edge_lag_days = self._build_masks_from_causal_spec(
            latent_names, manifest_cols, n_latent, n_manifest
        )
        # Store for builder lifecycle (build_model flow)
        self._edge_lag_days = edge_lag_days

        # Enable off-diagonal diffusion when correlation parameters exist
        # (marginalized confounders induce correlated process noise)
        has_correlation = any(p.role == ParameterRole.CORRELATION for p in model_spec.parameters)
        diffusion_mode: str = "free" if has_correlation else "diag"

        spec = SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=lambda_mat,
            drift="free",
            diffusion=diffusion_mode,
            cint="free",  # Enable CINT for non-zero asymptotic means
            manifest_means=None,  # Will be zeros
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

    def _build_masks_from_causal_spec(
        self,
        latent_names: list[str] | None,
        manifest_cols: list[str],
        n_latent: int,
        n_manifest: int,
    ) -> tuple[np.ndarray | None, jnp.ndarray, np.ndarray | None, dict[tuple[int, int], float]]:
        """Build drift_mask and lambda_mask from CausalSpec.

        Args:
            latent_names: Latent construct names (from AR params)
            manifest_cols: Manifest column names (from likelihoods)
            n_latent: Number of latent variables
            n_manifest: Number of manifest variables

        Returns:
            (drift_mask, lambda_mat, lambda_mask, edge_lag_days) — masks are
            None when causal_spec is not available (backward-compatible).
            edge_lag_days maps (effect_idx, cause_idx) → lag in days.
        """
        if self._causal_spec is None or latent_names is None:
            return None, jnp.eye(n_manifest, n_latent), None, {}

        causal_spec = self._causal_spec
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

        # Build name-to-index maps
        latent_idx = {name: i for i, name in enumerate(latent_names)}

        # Build construct lookup for lag_hours computation
        constructs = get_constructs(causal_spec)
        construct_map: dict[str, dict | Any] = {}
        for c in constructs:
            cname: str = c["name"] if isinstance(c, dict) else c.name
            construct_map[cname] = c

        # --- Drift mask ---
        # Diagonal always True (AR effects); off-diagonal True only where
        # a CausalEdge exists between two constructs.
        drift_mask = np.eye(n_latent, dtype=bool)
        # Accumulate edge metadata: (effect_idx, cause_idx) → lag_days
        edge_lag_days: dict[tuple[int, int], float] = {}
        for edge in edges:
            cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
            effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
            if cause in latent_idx and effect in latent_idx:
                ei, ci = latent_idx[effect], latent_idx[cause]
                # drift[effect_idx, cause_idx] = True (effect row, cause col)
                drift_mask[ei, ci] = True

                # Compute and store lag_hours for this edge
                lagged = edge.get("lagged", True) if isinstance(edge, dict) else edge.lagged
                dt_days = self._get_construct_dt_days("")  # model_clock is global
                lag_hours = (dt_days * 24.0) if lagged else 0
                if lag_hours > 0:
                    edge_lag_days[(ei, ci)] = lag_hours / 24.0

        # --- Lambda mask ---
        # Build from measurement model indicators → construct mapping.
        # First indicator per construct: fixed at 1.0 (reference indicator).
        # Additional indicators: free to sample (lambda_mask True).
        manifest_idx = {name: i for i, name in enumerate(manifest_cols)}
        lambda_mat_np = np.zeros((n_manifest, n_latent), dtype=np.float64)
        lambda_mask = np.zeros((n_manifest, n_latent), dtype=bool)

        # Track which constructs already have a reference indicator
        reference_set: set[str] = set()
        matched_manifests: set[str] = set()

        for indicator in indicators:
            ind_name = indicator.get("name") if isinstance(indicator, dict) else indicator.name
            construct = (
                indicator.get("construct_name")
                if isinstance(indicator, dict)
                else indicator.construct_name
            )

            if ind_name not in manifest_idx:
                continue
            if construct not in latent_idx:
                raise ValueError(
                    "CausalSpec measurement indicator references unknown construct: "
                    f"{ind_name!r} -> {construct!r}"
                )

            mi = manifest_idx[ind_name]
            li = latent_idx[construct]
            matched_manifests.add(ind_name)

            if construct not in reference_set:
                # First indicator for this construct: fixed reference
                lambda_mat_np[mi, li] = 1.0
                reference_set.add(construct)
            else:
                # Additional indicator: free to sample
                lambda_mask[mi, li] = True

        lambda_mat = jnp.array(lambda_mat_np)

        unmatched_manifests = sorted(set(manifest_cols) - matched_manifests)
        if unmatched_manifests:
            raise ValueError(
                "ModelSpec likelihoods could not be mapped to causal_spec measurement indicators: "
                f"{unmatched_manifests}"
            )

        return drift_mask, lambda_mat, lambda_mask, edge_lag_days

    def _convert_priors_to_ssm(
        self,
        priors: dict[str, dict],
        model_spec: ModelSpec | dict | None,
        ssm_spec: SSMSpec | None = None,
        edge_lag_days: dict[tuple[int, int], float] | None = None,
    ) -> tuple[
        SSMPriors,
        tuple[
            dict[str, tuple[str, int]],
            dict[str, tuple[str, int]],
            dict[str, tuple[str, int]],
            dict[str, tuple[str, int]],
            dict[str, tuple[str, int]],
        ],
    ]:
        """Convert prior proposals to SSMPriors.

        Uses ParameterRole from ModelSpec to determine which SSMPriors field
        each prior maps to, then normalizes distribution-specific params
        (Beta alpha/beta, Uniform lower/upper) to the mu/sigma format
        that SSMPriors expects.

        When ssm_spec has drift_mask or lambda_mask, builds per-element
        prior arrays that align with mask positions in row-major order.

        Falls back to keyword matching when ModelSpec is not available.

        Args:
            priors: Prior proposals from workers
            model_spec: Model specification for context (optional)
            ssm_spec: SSMSpec for per-element prior positioning (optional)
            edge_lag_days: Pre-computed edge lag metadata from mask building (optional)

        Returns:
            (SSMPriors, index_maps) — the priors and the 5-tuple of index maps
            from ``_build_prior_index_maps`` for downstream reuse.
        """
        ssm_priors = SSMPriors()

        # Build role lookup from ModelSpec if available
        role_by_name: dict[str, ParameterRole] = {}
        if model_spec:
            if isinstance(model_spec, dict) and model_spec.get("parameters"):
                spec_obj = ModelSpec.model_validate(model_spec)
            elif isinstance(model_spec, ModelSpec):
                spec_obj = model_spec
            else:
                spec_obj = None

            if spec_obj:
                for p in spec_obj.parameters:
                    role_by_name[p.name] = p.role

        # Collect per-element entries for array-valued priors
        # Maps SSMPriors field -> list of (array_index, normalized_dict)
        per_element: dict[str, list[tuple[int, dict]]] = {}

        # Build index maps from masks if available
        (
            offdiag_param_index,
            lambda_param_index,
            diag_param_index,
            diffusion_diag_param_index,
            diffusion_offdiag_param_index,
        ) = self._build_prior_index_maps(ssm_spec, model_spec)

        for param_name, prior_spec in priors.items():
            distribution = prior_spec.get("distribution", "Normal")
            params = prior_spec.get("params", {})

            # Normalize distribution params to mu/sigma
            normalized = _normalize_prior_params(distribution, params)

            # AR coefficient → apply DT-to-CT drift transform
            if param_name in diag_param_index:
                attr, idx = diag_param_index[param_name]
                construct_name = param_name.removeprefix("rho_").removeprefix("ar_")
                # Precedence: reference_interval_days > model_clock > default 1.0
                ref_days = prior_spec.get("reference_interval_days")
                if ref_days is not None and ref_days > 0:
                    dt = float(ref_days)
                else:
                    dt = self._get_construct_dt_days(construct_name)
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
                        f"AR prior '{param_name}' must have DT persistence mean in (0, 1), "
                        f"got {mu_ar:.3g}"
                    )
                mu_ar = min(max(mu_ar, 0.001), 0.999)
                sigma_ar = normalized.get("sigma", 0.2)
                mu_drift = -math.log(mu_ar) / dt
                sigma_drift = sigma_ar / (mu_ar * dt)  # delta method
                per_element.setdefault(attr, []).append(
                    (idx, {"mu": mu_drift, "sigma": sigma_drift})
                )
                continue

            # Fixed effect (beta) → apply DT-to-CT coupling rate transform
            # Literature betas are discrete-time cross-lagged coefficients;
            # the drift off-diagonal is a continuous-time rate: β_CT ≈ β_DT / dt
            if param_name in offdiag_param_index:
                attr, idx = offdiag_param_index[param_name]
                # Precedence: reference_interval_days > model_clock > default 1.0
                ref_days = prior_spec.get("reference_interval_days")
                if ref_days is not None and ref_days > 0:
                    dt = float(ref_days)
                else:
                    dt = 1.0  # default daily
                    # Parse "beta_<cause>_<effect>" to get effect construct's dt
                    if ssm_spec and ssm_spec.latent_names:
                        latent_set = set(ssm_spec.latent_names)
                        compound = param_name.removeprefix("beta_")
                        split = _split_compound_name(compound, latent_set, latent_set)
                        if split:
                            _cause, effect = split
                            dt = self._get_construct_dt_days(effect)
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
            # Correlation → diffusion off-diagonal (no DT-to-CT transform needed;
            # diffusion Cholesky elements are already continuous-time)
            if param_name in diffusion_offdiag_param_index:
                attr, idx = diffusion_offdiag_param_index[param_name]
                per_element.setdefault(attr, []).append((idx, normalized))
                continue

            # Determine SSMPriors field via role (preferred) or keyword fallback
            role = role_by_name.get(param_name)
            if role and role in _ROLE_TO_SSM:
                attr, defaults = _ROLE_TO_SSM[role]
                # Merge normalized params with defaults (normalized takes priority)
                merged = {k: normalized.get(k, v) for k, v in defaults.items()}
                setattr(ssm_priors, attr, merged)
            else:
                # Keyword fallback for when no ModelSpec role is available
                name_lower = param_name.lower()
                matched = False
                for keywords, attr, defaults in _KEYWORD_RULES:
                    matching_kw = [kw for kw in keywords if kw in name_lower]
                    if matching_kw:
                        logger.debug(
                            "Prior '%s': keyword fallback matched '%s' -> %s",
                            param_name,
                            matching_kw[0],
                            attr,
                        )
                        merged = {k: normalized.get(k, v) for k, v in defaults.items()}
                        setattr(ssm_priors, attr, merged)
                        matched = True
                        break
                if not matched:
                    logger.debug(
                        "Prior '%s': no role or keyword match found, skipping",
                        param_name,
                    )

        # Build array-valued priors from per-element entries
        for attr, entries in per_element.items():
            current = getattr(ssm_priors, attr)
            expected_size = self._expected_prior_size(attr, ssm_spec)
            n_total = max(idx for idx, _ in entries) + 1
            if expected_size is not None:
                n_total = max(n_total, expected_size)

            # Build arrays from defaults + positioned entries
            include_mu = "mu" in current or any("mu" in normed for _, normed in entries)
            include_sigma = "sigma" in current or any("sigma" in normed for _, normed in entries)

            mu_arr = [float(current.get("mu", 0.0))] * n_total if include_mu else None
            sigma_arr = [float(current.get("sigma", 0.5))] * n_total if include_sigma else None

            for idx, normed in entries:
                if "mu" in normed and mu_arr is not None:
                    mu_arr[idx] = float(normed["mu"])
                if "sigma" in normed and sigma_arr is not None:
                    sigma_arr[idx] = float(normed["sigma"])

            result: dict[str, list[float]] = {}
            if mu_arr is not None:
                result["mu"] = mu_arr
            if sigma_arr is not None:
                result["sigma"] = sigma_arr

            # Propagate bounds if any entry has them
            has_bounds = any("lower" in n for _, n in entries)
            if has_bounds:
                lower_arr = [-1e6] * n_total
                upper_arr = [1e6] * n_total
                for idx, normed in entries:
                    lower_arr[idx] = float(normed.get("lower", -1e6))
                    upper_arr[idx] = float(normed.get("upper", 1e6))
                result["lower"] = lower_arr
                result["upper"] = upper_arr

            setattr(ssm_priors, attr, result)

        # The compiled prior family is factorized across parameters, so keep the
        # DT→CT transform element-wise rather than applying a mean-only joint logm rewrite.
        self._warn_first_order_approximation(ssm_priors)

        # Check consistency between CT drift rates and edge lag_hours
        if ssm_spec:
            self._check_drift_lag_consistency(ssm_priors, ssm_spec, edge_lag_days=edge_lag_days)

        index_maps = (
            offdiag_param_index,
            lambda_param_index,
            diag_param_index,
            diffusion_diag_param_index,
            diffusion_offdiag_param_index,
        )
        return ssm_priors, index_maps

    @staticmethod
    def _warn_first_order_approximation(ssm_priors: SSMPriors) -> None:
        """Log warning when off-diagonal drift magnitudes suggest first-order error > 20%.

        The first-order approximation beta_CT = beta_DT / dt has error
        O(dt * ||A_offdiag||). When any off-diagonal magnitude exceeds 20%
        of the corresponding diagonal magnitude, the approximation may be
        significantly inaccurate.
        """
        diag_prior = ssm_priors.drift_diag
        offdiag_prior = ssm_priors.drift_offdiag
        if diag_prior is None or offdiag_prior is None:
            return

        diag_mu = diag_prior.get("mu")
        offdiag_mu = offdiag_prior.get("mu")
        if diag_mu is None or offdiag_mu is None:
            return

        # Normalize to lists
        if isinstance(diag_mu, (int, float)):
            diag_mu = [diag_mu]
        if isinstance(offdiag_mu, (int, float)):
            offdiag_mu = [offdiag_mu]

        if not diag_mu or not offdiag_mu:
            return

        # Use minimum diagonal magnitude as reference
        min_diag = min(abs(float(d)) for d in diag_mu)
        if min_diag < NUMERICAL_EPSILON:
            return

        for i, od in enumerate(offdiag_mu):
            ratio = abs(float(od)) / min_diag
            if ratio > 0.2:
                logger.warning(
                    "First-order DT->CT approximation may be inaccurate: "
                    "off-diagonal drift[%d] magnitude (%.3f) is %.0f%% of "
                    "minimum diagonal magnitude (%.3f). Consider a shorter "
                    "reference interval or eliciting priors directly on CT rates.",
                    i,
                    abs(float(od)),
                    ratio * 100,
                    min_diag,
                )
                break  # One warning is enough

    def _check_drift_lag_consistency(
        self,
        ssm_priors: SSMPriors,
        ssm_spec: SSMSpec,
        edge_lag_days: dict[tuple[int, int], float] | None = None,
    ) -> None:
        """Check CT drift rates against expected lag from causal edge metadata.

        For each off-diagonal drift entry with a known edge lag, compares
        the implied coupling timescale (1/|A[i,j]|) with the expected lag.
        Logs a warning when they differ by more than 5x, suggesting the
        literature prior may be calibrated to a different timescale than
        the causal model expects.
        """
        edge_lags = (
            edge_lag_days if edge_lag_days is not None else getattr(self, "_edge_lag_days", {})
        )
        if not edge_lags:
            return

        offdiag_prior = ssm_priors.drift_offdiag
        if offdiag_prior is None or "mu" not in offdiag_prior:
            return

        mu_arr = offdiag_prior["mu"]
        if not isinstance(mu_arr, list):
            return

        n = ssm_spec.n_latent
        # Build off-diagonal position map (same order as drift mask iteration)
        offdiag_positions: list[tuple[int, int]] = []
        if ssm_spec.drift_mask is not None:
            for i in range(n):
                for j in range(n):
                    if i != j and ssm_spec.drift_mask[i, j]:
                        offdiag_positions.append((i, j))

        for flat_idx, (ei, ci) in enumerate(offdiag_positions):
            if flat_idx >= len(mu_arr):
                break
            if (ei, ci) not in edge_lags:
                continue

            mu_ct = abs(float(mu_arr[flat_idx]))
            if mu_ct < NUMERICAL_EPSILON:
                continue

            expected_lag_days = edge_lags[(ei, ci)]
            implied_timescale_days = 1.0 / mu_ct

            ratio = max(implied_timescale_days, expected_lag_days) / max(
                min(implied_timescale_days, expected_lag_days), NUMERICAL_EPSILON
            )
            if ratio > 5.0:
                cause_name = ssm_spec.latent_names[ci] if ssm_spec.latent_names else f"latent_{ci}"
                effect_name = ssm_spec.latent_names[ei] if ssm_spec.latent_names else f"latent_{ei}"
                logger.warning(
                    "Drift rate for %s->%s implies timescale %.1f days, "
                    "but edge lag suggests %.1f days (%.0fx mismatch). "
                    "The literature prior may be calibrated to a different "
                    "observation interval than the causal model expects.",
                    cause_name,
                    effect_name,
                    implied_timescale_days,
                    expected_lag_days,
                    ratio,
                )

    def _build_prior_index_maps(
        self,
        ssm_spec: SSMSpec | None,
        model_spec: ModelSpec | dict | None,
    ) -> tuple[
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
    ]:
        """Build parameter name → (SSMPriors field, array index) maps.

        Uses drift_mask and lambda_mask to determine which array position
        each causal parameter occupies, so per-element priors align with
        the sampling order in _sample_drift/_sample_lambda/_sample_diffusion.

        Returns:
            (offdiag_param_index, lambda_param_index, diag_param_index,
             diffusion_diag_param_index, diffusion_offdiag_param_index) —
            all are {param_name: (ssm_field, index)} dicts. Empty if no
            spec/masks.
        """
        offdiag_index: dict[str, tuple[str, int]] = {}
        lambda_index: dict[str, tuple[str, int]] = {}
        diag_index: dict[str, tuple[str, int]] = {}
        diffusion_diag_index: dict[str, tuple[str, int]] = {}
        diffusion_offdiag_index: dict[str, tuple[str, int]] = {}

        if ssm_spec is None:
            return (
                offdiag_index,
                lambda_index,
                diag_index,
                diffusion_diag_index,
                diffusion_offdiag_index,
            )

        # Parse model_spec for parameter names + roles
        if not model_spec:
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
        latent_idx_map = {name: i for i, name in enumerate(latent_names)}
        strict_structure = self._causal_spec is not None

        # --- Drift diagonal index (AR coefficients) ---
        for p in spec_obj.parameters:
            if p.role != ParameterRole.AR_COEFFICIENT:
                continue
            # Convention: parameter name "rho_<construct>" or "ar_<construct>"
            construct = p.name.removeprefix("rho_").removeprefix("ar_")
            if construct in latent_idx_map:
                diag_index[p.name] = ("drift_diag", latent_idx_map[construct])
            elif strict_structure:
                raise ValueError(
                    "AR parameter does not reference a construct in causal_spec: "
                    f"{p.name!r} not in {sorted(latent_idx_map)}"
                )

        # --- Diffusion diagonal index (residual SDs) ---
        for p in spec_obj.parameters:
            if p.role != ParameterRole.RESIDUAL_SD:
                continue
            construct = p.name.removeprefix("sigma_")
            if construct in latent_idx_map:
                diffusion_diag_index[p.name] = ("diffusion_diag", latent_idx_map[construct])
            elif strict_structure:
                raise ValueError(
                    "RESIDUAL_SD parameter does not reference a construct in causal_spec: "
                    f"{p.name!r} not in {sorted(latent_idx_map)}"
                )

        # --- Drift off-diagonal index ---
        if ssm_spec.drift_mask is not None:
            n = ssm_spec.n_latent
            # Build ordered list of (i, j) positions matching _sample_drift
            positions = []
            for i in range(n):
                for j in range(n):
                    if i != j and ssm_spec.drift_mask[i, j]:
                        positions.append((i, j))

            # Map FIXED_EFFECT parameters to positions via cause→effect naming
            latent_name_set = set(latent_idx_map.keys())
            for p in spec_obj.parameters:
                if p.role != ParameterRole.FIXED_EFFECT:
                    continue
                # Convention: parameter name "beta_<cause>_<effect>"
                compound = p.name.removeprefix("beta_")
                result = _split_compound_name(compound, latent_name_set, latent_name_set)
                if result is None:
                    message = (
                        "Could not parse FIXED_EFFECT parameter "
                        f"{p.name!r} into (cause, effect) from known latents "
                        f"{sorted(latent_name_set)}"
                    )
                    if strict_structure:
                        raise ValueError(message)
                    logger.warning("%s", message)
                    continue
                cause_name, effect_name = result
                pos = (latent_idx_map[effect_name], latent_idx_map[cause_name])
                if pos in positions:
                    offdiag_index[p.name] = ("drift_offdiag", positions.index(pos))
                elif strict_structure:
                    raise ValueError(
                        "FIXED_EFFECT parameter does not correspond to an edge in causal_spec: "
                        f"{p.name!r}"
                    )

        # --- Lambda free index ---
        if ssm_spec.lambda_mask is not None:
            manifest_names = ssm_spec.manifest_names or []
            manifest_idx_map = {name: i for i, name in enumerate(manifest_names)}

            # Build ordered list matching _sample_lambda
            positions = []
            for i in range(ssm_spec.n_manifest):
                for j in range(ssm_spec.n_latent):
                    if ssm_spec.lambda_mask[i, j]:
                        positions.append((i, j))

            manifest_name_set = set(manifest_idx_map.keys())
            for p in spec_obj.parameters:
                if p.role != ParameterRole.LOADING:
                    continue
                # Convention: parameter name "lambda_<indicator>_<construct>"
                compound = p.name.removeprefix("lambda_")
                result = _split_compound_name(compound, manifest_name_set, latent_name_set)
                if result is None:
                    message = (
                        "Could not parse LOADING parameter "
                        f"{p.name!r} into (indicator, construct) from known "
                        f"manifests {sorted(manifest_name_set)} / latents {sorted(latent_name_set)}"
                    )
                    if strict_structure:
                        raise ValueError(message)
                    logger.warning("%s", message)
                    continue
                ind_name, construct_name = result
                pos = (manifest_idx_map[ind_name], latent_idx_map[construct_name])
                if pos in positions:
                    lambda_index[p.name] = ("lambda_free", positions.index(pos))
                elif strict_structure:
                    raise ValueError(
                        "LOADING parameter does not correspond to a free loading in causal_spec: "
                        f"{p.name!r}"
                    )

        # --- Diffusion off-diagonal index (correlation parameters) ---
        # Lower-triangular positions matching _sample_diffusion ordering:
        # for i in range(n): for j in range(i): position (i, j)
        if ssm_spec.diffusion == "free":
            n = ssm_spec.n_latent
            lower_positions: list[tuple[int, int]] = []
            for i in range(n):
                for j in range(i):
                    lower_positions.append((i, j))

            latent_name_set = set(latent_idx_map.keys())
            for p in spec_obj.parameters:
                if p.role != ParameterRole.CORRELATION:
                    continue
                # Convention: parameter name "cor_<state1>_<state2>"
                compound = p.name.removeprefix("cor_")
                result = _split_compound_name(compound, latent_name_set, latent_name_set)
                if result is None:
                    message = (
                        "Could not parse CORRELATION parameter "
                        f"{p.name!r} into (state1, state2) from known latents "
                        f"{sorted(latent_name_set)}"
                    )
                    if strict_structure:
                        raise ValueError(message)
                    logger.warning("%s", message)
                    continue
                s1_name, s2_name = result
                idx1, idx2 = latent_idx_map[s1_name], latent_idx_map[s2_name]
                # Lower-triangular: larger index first
                pos = (max(idx1, idx2), min(idx1, idx2))
                if pos in lower_positions:
                    diffusion_offdiag_index[p.name] = (
                        "diffusion_offdiag",
                        lower_positions.index(pos),
                    )
                elif strict_structure:
                    raise ValueError(
                        "CORRELATION parameter does not correspond to a modeled latent pair: "
                        f"{p.name!r}"
                    )

        return (
            offdiag_index,
            lambda_index,
            diag_index,
            diffusion_diag_index,
            diffusion_offdiag_index,
        )

    def _compile_parameter_bindings(
        self,
        ssm_spec: SSMSpec,
        model_spec: ModelSpec | dict | None,
        index_maps: tuple[
            dict[str, tuple[str, int]],
            dict[str, tuple[str, int]],
            dict[str, tuple[str, int]],
            dict[str, tuple[str, int]],
            dict[str, tuple[str, int]],
        ]
        | None = None,
    ) -> list[dict[str, Any]]:
        """Compile semantic parameter bindings to NumPyro sample sites."""
        if index_maps is not None:
            (
                offdiag_index,
                lambda_index,
                diag_index,
                diffusion_diag_index,
                diffusion_offdiag_index,
            ) = index_maps
        else:
            (
                offdiag_index,
                lambda_index,
                diag_index,
                diffusion_diag_index,
                diffusion_offdiag_index,
            ) = self._build_prior_index_maps(ssm_spec, model_spec)

        bindings: list[dict[str, Any]] = []
        index_maps = (
            diag_index,
            offdiag_index,
            diffusion_diag_index,
            diffusion_offdiag_index,
            lambda_index,
        )
        for mapping in index_maps:
            for param_name, (prior_field, flat_index) in sorted(mapping.items()):
                sample_site = _SAMPLE_SITE_FOR_PRIOR_FIELD.get(prior_field)
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

    def build_model(
        self,
        X: pl.DataFrame,
        y: np.ndarray | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> SSMModel:
        """Build the NumPyro SSM model.

        Args:
            X: Polars DataFrame with indicator columns, time, and optional subject_id
            y: Optional target (if not in X)

        Returns:
            The constructed SSMModel
        """
        # Determine specification
        if self._ssm_spec is None and self._model_spec is None:
            # Auto-detect from data
            manifest_cols = _default_manifest_columns(X)
            spec = SSMSpec(
                n_latent=len(manifest_cols),
                n_manifest=len(manifest_cols),
                lambda_mat=jnp.eye(len(manifest_cols)),
            )
            raw_priors: dict[str, dict] = {
                k: v.model_dump() if isinstance(v, PriorProposal) else v
                for k, v in self._priors.items()
            }
            priors, _index_maps = self._convert_priors_to_ssm(
                raw_priors, self._model_spec or {}, ssm_spec=spec
            )
        else:
            spec, priors = self.compile_inputs()

        spec = self._hydrate_discrete_manifest_metadata(spec, X)
        self._validate_observation_support(spec, X)

        # Create model with PF config from model_config
        n_particles = self._model_config.get("n_particles", 200)
        pf_seed = self._model_config.get("pf_seed", 0)
        self._model = SSMModel(spec, priors, n_particles=n_particles, pf_seed=pf_seed)
        self._model.parameter_bindings = list(self._parameter_bindings or [])
        self._spec = spec

        return self._model

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Fit the SSM model to data.

        Args:
            X: Polars DataFrame with indicator columns, time, and optional subject_id
            y: Optional target (if not in X)
            **kwargs: Additional arguments passed to inference

        Returns:
            InferenceResult with posterior samples
        """
        if self._model is None:
            self.build_model(X, y)

        assert self._model is not None

        # Prepare data
        observations, times = self._prepare_data(X)

        # Merge sampler config with kwargs
        sampler_config = {**self._sampler_config, **kwargs}

        # Extract method (default to auto = structural routing) without mutating
        method = sampler_config.get("method", "auto")
        fit_kwargs = {k: v for k, v in sampler_config.items() if k != "method"}

        result = fit(
            self._model,
            observations=observations,
            times=times,
            method=method,
            **fit_kwargs,
        )

        self._result = result
        return result

    def _hydrate_discrete_manifest_metadata(self, spec: SSMSpec, X: pl.DataFrame) -> SSMSpec:
        """Infer per-channel discrete level counts from encoded wide data."""
        manifest_cols, manifest_dists = _resolve_manifest_metadata(spec, X)
        needs_levels = any(
            dist in (DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL)
            for dist in manifest_dists
        )
        if not needs_levels:
            return spec

        inferred_counts = [0] * spec.n_manifest
        for idx, (column, dist) in enumerate(zip(manifest_cols, manifest_dists, strict=False)):
            if dist not in (DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL):
                continue

            values = (
                X.select(column)
                .drop_nulls()
                .to_series()
                .cast(pl.Float64, strict=False)
                .drop_nulls()
                .to_numpy()
            )
            if values.size == 0:
                raise ValueError(
                    f"Indicator '{column}' uses discrete emission '{dist.value}' but has no data"
                )

            rounded = np.rint(values)
            if not np.allclose(values, rounded, atol=1e-6):
                raise ValueError(
                    f"Indicator '{column}' uses discrete emission '{dist.value}' but data are not "
                    "integer-encoded"
                )

            unique_levels = sorted({int(v) for v in rounded.tolist()})
            if unique_levels[0] != 0 or unique_levels != list(range(unique_levels[-1] + 1)):
                raise ValueError(
                    f"Indicator '{column}' uses discrete emission '{dist.value}' but encoded levels "
                    f"are not contiguous from 0: {unique_levels}"
                )
            if len(unique_levels) < 2:
                raise ValueError(
                    f"Indicator '{column}' uses discrete emission '{dist.value}' but only "
                    f"{len(unique_levels)} level(s) are present"
                )
            inferred_counts[idx] = len(unique_levels)

        if spec.manifest_level_counts is None:
            return replace(spec, manifest_level_counts=inferred_counts)

        if len(spec.manifest_level_counts) != spec.n_manifest:
            raise ValueError(
                "SSMSpec manifest_level_counts length does not match n_manifest: "
                f"{len(spec.manifest_level_counts)} vs {spec.n_manifest}"
            )

        resolved_counts = list(spec.manifest_level_counts)
        for idx, inferred_count in enumerate(inferred_counts):
            if inferred_count == 0:
                resolved_counts[idx] = 0
                continue
            existing_count = resolved_counts[idx]
            if existing_count in (0, inferred_count):
                resolved_counts[idx] = inferred_count
                continue
            raise ValueError(
                "Discrete level metadata mismatch for "
                f"'{manifest_cols[idx]}': spec={existing_count}, data={inferred_count}"
            )

        return replace(spec, manifest_level_counts=resolved_counts)

    def _validate_observation_support(self, spec: SSMSpec, X: Any) -> None:
        """Reject likelihoods whose support is incompatible with observed data."""
        manifest_cols, manifest_dists = _resolve_manifest_metadata(spec, X)

        issues: list[str] = []
        for column, dist in zip(manifest_cols, manifest_dists, strict=False):
            values = _extract_numeric_column_values(X, column)
            if values.size == 0:
                continue
            if np.any(~np.isfinite(values)):
                issues.append(
                    f"- '{column}' uses {dist.value} emission but observed data contain "
                    "non-finite values"
                )
                continue

            invalid = np.zeros(values.shape, dtype=bool)
            support = ""
            if dist == DistributionFamily.GAMMA:
                invalid = values <= 0.0
                support = "gamma requires y > 0"
            elif dist == DistributionFamily.BETA:
                invalid = (values <= 0.0) | (values >= 1.0)
                support = "beta requires 0 < y < 1"
            elif dist in (DistributionFamily.POISSON, DistributionFamily.NEGATIVE_BINOMIAL):
                rounded = np.rint(values)
                invalid = (values < 0.0) | (~np.isclose(values, rounded, atol=1e-6))
                support = f"{dist.value} requires non-negative integer counts"
            elif dist == DistributionFamily.BERNOULLI:
                invalid = ~np.isin(values, [0.0, 1.0])
                support = "bernoulli requires binary values in {0, 1}"
            elif dist in (DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL):
                rounded = np.rint(values)
                invalid = (values < 0.0) | (~np.isclose(values, rounded, atol=1e-6))
                support = f"{dist.value} requires non-negative integer-encoded levels"

            if not np.any(invalid):
                continue

            bad_values = values[invalid]
            issues.append(
                f"- '{column}' uses {dist.value} emission but {bad_values.size}/{values.size} "
                f"observations are outside support ({support}; "
                f"min={float(values.min()):.3g}, max={float(values.max()):.3g})"
            )

        if issues:
            raise ValueError("Observation support check failed:\n" + "\n".join(issues))

    def _prepare_data(self, X: pl.DataFrame) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare data for SSM fitting.

        Expects wide-format data from pivot_to_wide() which already converts
        datetimes to fractional days. The 'time' column should be numeric.

        Args:
            X: Polars DataFrame with observations (wide format)

        Returns:
            Tuple of (observations, times)
        """
        # Get manifest columns
        if self._spec is not None and self._spec.manifest_names:
            manifest_cols = self._spec.manifest_names
        else:
            manifest_cols = _default_manifest_columns(X)

        # Extract observations
        observations = jnp.array(X.select(manifest_cols).to_numpy(), dtype=jnp.float32)

        # Extract times (already fractional days from pivot_to_wide)
        time_col = "time" if "time" in X.columns else "time_bucket"
        if time_col in X.columns:
            times = jnp.array(X[time_col].to_numpy(), dtype=jnp.float32)
        else:
            # Default: integer sequence
            times = jnp.arange(X.height, dtype=jnp.float32)

        return observations, times

    def sample_prior_predictive(self, samples: int = 500, times: jnp.ndarray | None = None) -> Any:
        """Sample from the prior predictive distribution.

        Args:
            samples: Number of samples
            times: Optional time points; defaults to arange(10)

        Returns:
            Prior predictive samples
        """
        if times is None:
            times = jnp.arange(10, dtype=jnp.float32)

        spec = self._spec
        if spec is None:
            if self._model is not None:
                spec = self._model.spec
            elif self._ssm_spec is not None:
                spec = self._ssm_spec
            elif self._model_spec is not None:
                spec, _priors = self.compile_inputs()
            else:
                raise ValueError("Cannot sample prior predictive without an SSM specification")

        from causal_ssm_agent.models.ssm.prior_predictive_runtime import (
            sample_prior_predictive_from_compiled_semantics,
            sample_prior_predictive_from_priors,
        )
        from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

        manifest_dists = spec.manifest_dists or [spec.manifest_dist] * spec.n_manifest
        needs_hydration = any(
            dist in (DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL)
            for dist in manifest_dists
        )
        if needs_hydration and spec.manifest_level_counts is None:
            raise ValueError(
                "Prior predictive for ordered/categorical emissions requires hydrated "
                "manifest_level_counts. Build the model with data first."
            )

        if self._compiled_prior_semantics is not None and self._spec is None:
            return sample_prior_predictive_from_compiled_semantics(
                spec,
                self._compiled_prior_semantics,
                times,
                num_samples=samples,
            )

        compiled_spec, priors = self.compile_inputs()
        runtime_spec = self._spec if self._spec is not None else compiled_spec
        return sample_prior_predictive_from_priors(
            runtime_spec,
            priors,
            times,
            num_samples=samples,
        )

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Get posterior samples.

        Returns:
            Dict of posterior samples
        """
        if self._result is not None:
            return self._result.get_samples()
        raise ValueError("Model must be fit before getting samples")

    def summary(self) -> pl.DataFrame:
        """Get summary statistics for posterior.

        Returns:
            Polars DataFrame with summary statistics
        """
        if self._result is None:
            raise ValueError("Model must be fit before getting summary")

        self._result.print_summary()

        # Also return as DataFrame
        samples = self.get_samples()
        summary_data = []
        for name, values in samples.items():
            if values.ndim == 1:
                summary_data.append(
                    {
                        "parameter": name,
                        "mean": float(jnp.mean(values)),
                        "std": float(jnp.std(values)),
                        "5%": float(jnp.percentile(values, 5)),
                        "95%": float(jnp.percentile(values, 95)),
                    }
                )
        return pl.DataFrame(summary_data)


def translate_spec(
    model_spec: ModelSpec | dict,
    causal_spec: dict | None = None,
) -> tuple[SSMSpec, dict[tuple[int, int], float]]:
    """Pure function: convert ModelSpec to SSMSpec with edge lags.

    Returns (spec, edge_lag_days) with explicit data flow.
    """
    builder = SSMModelBuilder(model_spec=model_spec, causal_spec=causal_spec)
    spec, edge_lag_days = builder._convert_spec_to_ssm(model_spec)
    return spec, edge_lag_days


def compile_priors(
    raw_priors: dict[str, dict],
    model_spec: ModelSpec | dict,
    ssm_spec: SSMSpec,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
    causal_spec: dict | None = None,
) -> tuple[
    SSMPriors,
    tuple[
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
    ],
]:
    """Pure function: convert prior proposals to SSMPriors.

    Returns (ssm_priors, index_maps) with explicit data flow.
    """
    builder = SSMModelBuilder(model_spec=model_spec, priors=raw_priors, causal_spec=causal_spec)
    return builder._convert_priors_to_ssm(
        raw_priors, model_spec, ssm_spec=ssm_spec, edge_lag_days=edge_lag_days
    )


def compile_ssm_inputs(
    model_spec: ModelSpec | dict,
    priors: dict[str, dict],
    causal_spec: dict | None = None,
) -> tuple[SSMSpec, SSMPriors, list[dict[str, Any]]]:
    """Pure function: full compilation pipeline as function composition.

    Combines translate_spec -> compile_priors -> bind_parameters with
    explicit data flow (no hidden instance state).
    """
    spec, edge_lag_days = translate_spec(model_spec, causal_spec)
    ssm_priors, index_maps = compile_priors(priors, model_spec, spec, edge_lag_days, causal_spec)
    builder = SSMModelBuilder(model_spec=model_spec, causal_spec=causal_spec)
    bindings = builder._compile_parameter_bindings(spec, model_spec, index_maps=index_maps)
    return spec, ssm_priors, bindings


def build_ssm_builder(
    raw_data: pl.DataFrame,
    model_spec: ModelSpec | dict | None = None,
    priors: dict[str, PriorProposal] | dict[str, dict] | None = None,
    causal_spec: dict | None = None,
    sampler_config: dict | None = None,
    compiled_ssm: dict | None = None,
) -> SSMModelBuilder:
    """Single canonical entry point for constructing a ready-to-use SSMModelBuilder.

    Encapsulates the repeated pattern of:
        builder = SSMModelBuilder(...)
        X = pivot_to_wide(raw_data)
        builder.build_model(X)

    Args:
        model_spec: Model specification (dict or ModelSpec)
        priors: Prior proposals by parameter name
        raw_data: Raw timestamped data (long format)
        causal_spec: CausalSpec dict for DAG-constrained masks
        sampler_config: Override sampler configuration
        compiled_ssm: Serialized compiled artifact from stage 4

    Returns:
        A fully built SSMModelBuilder (model constructed, ready for fit/sample)

    Raises:
        ValueError: If raw_data is empty
    """
    if compiled_ssm is not None:
        from causal_ssm_agent.models.ssm_compiler import build_compiled_ssm_builder

        return build_compiled_ssm_builder(
            compiled_ssm,
            raw_data,
            sampler_config=sampler_config,
        )

    from causal_ssm_agent.utils.data import pivot_to_wide

    if raw_data.is_empty():
        raise ValueError("Cannot build SSM model from empty data")

    builder = SSMModelBuilder(
        model_spec=model_spec,
        priors=priors,
        causal_spec=causal_spec,
        sampler_config=sampler_config,
    )
    X = pivot_to_wide(raw_data)
    builder.build_model(X)
    return builder
