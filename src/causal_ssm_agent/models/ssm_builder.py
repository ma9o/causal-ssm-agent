"""SSM Model Builder for causal SSM pipeline integration.

Provides a model builder interface compatible with the causal SSM pipeline
while using the NumPyro SSM implementation underneath.
"""

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
import polars as pl

from causal_ssm_agent.models.ssm import (
    InferenceResult,
    SSMModel,
    SSMPriors,
    SSMSpec,
    fit,
)
from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionFamily,
    ModelSpec,
    ParameterRole,
    validate_model_spec,
)
from causal_ssm_agent.workers.schemas_prior import PriorProposal

logger = logging.getLogger(__name__)

# Distributions that have native emission functions in emissions.py.
_SUPPORTED_EMISSIONS: set[DistributionFamily] = {
    DistributionFamily.GAUSSIAN,
    DistributionFamily.STUDENT_T,
    DistributionFamily.POISSON,
    DistributionFamily.GAMMA,
    DistributionFamily.BERNOULLI,
    DistributionFamily.NEGATIVE_BINOMIAL,
    DistributionFamily.BETA,
}


# Map ParameterRole to SSMPriors field and default mu/sigma params.
# This replaces the old keyword-matching _PRIOR_RULES.
_ROLE_TO_SSM: dict[ParameterRole, tuple[str, dict]] = {
    ParameterRole.AR_COEFFICIENT: ("drift_diag", {"mu": -0.5, "sigma": 1.0}),
    ParameterRole.FIXED_EFFECT: ("drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    ParameterRole.RESIDUAL_SD: ("diffusion_diag", {"sigma": 1.0}),
    ParameterRole.LOADING: ("lambda_free", {"mu": 0.5, "sigma": 0.5}),
    ParameterRole.CORRELATION: ("drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
}

# Fallback keyword matching for parameters without a role in the ModelSpec
# (e.g. when priors are provided as a flat dict without ParameterSpec context)
_KEYWORD_RULES: list[tuple[list[str], str, dict]] = [
    (["rho", "ar"], "drift_diag", {"mu": -0.5, "sigma": 1.0}),
    (["beta"], "drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    (["sigma", "sd"], "diffusion_diag", {"sigma": 1.0}),
    (["lambda", "loading"], "lambda_free", {"mu": 0.5, "sigma": 0.5}),
    (["cor"], "drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
]


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
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        causal_spec: dict | None = None,
    ):
        """Initialize the SSM model builder.

        Args:
            model_spec: Model specification from orchestrator (will be converted)
            priors: Prior proposals for each parameter
            ssm_spec: Direct SSMSpec (overrides model_spec conversion)
            model_config: Override model configuration (n_particles, pf_seed)
            sampler_config: Override sampler configuration
            causal_spec: CausalSpec dict with latent model edges and measurement
                model indicators. When provided, _convert_spec_to_ssm builds
                drift_mask and lambda_mask from the DAG structure.
        """
        self._model_spec = model_spec
        self._priors = priors or {}
        self._ssm_spec = ssm_spec
        self._model_config = model_config or {}
        self._sampler_config = sampler_config or self.get_default_sampler_config()
        self._causal_spec = causal_spec

        self._model: SSMModel | None = None
        self._result: InferenceResult | None = None

    @staticmethod
    def get_default_sampler_config() -> dict:
        """Default sampler configuration, read from config.yaml."""
        from causal_ssm_agent.utils.config import get_config

        return get_config().inference.to_sampler_config()

    def _convert_spec_to_ssm(self, model_spec: ModelSpec | dict) -> SSMSpec:
        """Convert ModelSpec to SSMSpec.

        When causal_spec is provided, builds drift_mask and lambda_mask from
        the DAG structure instead of using a fully free drift and identity lambda.

        Args:
            model_spec: Model specification

        Returns:
            SSMSpec for continuous-time model
        """
        if isinstance(model_spec, dict):
            from causal_ssm_agent.orchestrator.schemas_model import ModelSpec

            model_spec = ModelSpec.model_validate(model_spec)

        issues = validate_model_spec(model_spec)
        for issue in issues:
            logger.warning(
                "ModelSpec %s: %s — %s", issue["severity"], issue["name"], issue["issue"]
            )

        # Extract dimensions from data
        manifest_cols = [lik.variable for lik in model_spec.likelihoods]
        n_manifest = len(manifest_cols)

        # Infer latent structure from parameters
        # Look for AR coefficients to determine number of latent processes
        ar_params = [p for p in model_spec.parameters if p.role == ParameterRole.AR_COEFFICIENT]
        n_latent = max(len(ar_params), 1)

        if not ar_params:
            logger.warning(
                "No AR_COEFFICIENT parameters found in ModelSpec; falling back to n_latent=1. "
                "Set AR coefficients explicitly for multi-latent models."
            )

        if n_manifest < n_latent:
            logger.warning(
                "n_manifest (%d) < n_latent (%d): lambda matrix may be rank-deficient",
                n_manifest,
                n_latent,
            )

        # Determine per-indicator noise families from likelihoods.
        # Distributions are passed through directly — no approximation.
        manifest_dists: list[DistributionFamily] = []
        for lik in model_spec.likelihoods:
            if lik.distribution not in _SUPPORTED_EMISSIONS:
                raise ValueError(
                    f"Indicator '{lik.variable}': distribution '{lik.distribution}' "
                    f"has no native emission function. Supported: "
                    f"{sorted(d.value for d in _SUPPORTED_EMISSIONS)}."
                )
            manifest_dists.append(lik.distribution)

        # Scalar fallback: first non-Gaussian type (for PF dispatch)
        manifest_dist = DistributionFamily.GAUSSIAN
        for nd in manifest_dists:
            if nd != DistributionFamily.GAUSSIAN:
                manifest_dist = nd
                break

        # Derive latent names from AR parameter names (e.g. rho_X → X)
        latent_names = [p.name.removeprefix("rho_") for p in ar_params] if ar_params else None

        # Include time-invariant constructs from causal_spec as additional latents.
        # They get near-zero drift/diffusion (quasi-constant, determined by initial state).
        time_invariant_mask = None
        if latent_names is not None and self._causal_spec is not None:
            latent_data = self._causal_spec.get("latent", {})
            constructs = latent_data.get("constructs", [])
            tv_set = set(latent_names)
            ti_names = []
            for c in constructs:
                name = c.get("name") if isinstance(c, dict) else c.name
                temporal = c.get("temporal_status") if isinstance(c, dict) else c.temporal_status
                if temporal == "time_invariant" and name not in tv_set:
                    ti_names.append(name)
            if ti_names:
                logger.info(
                    "Including %d time-invariant constructs as quasi-constant latents: %s",
                    len(ti_names),
                    ti_names,
                )
                time_invariant_mask = np.array([False] * len(latent_names) + [True] * len(ti_names))
                latent_names = latent_names + ti_names
                n_latent = len(latent_names)

        # Build masks from causal_spec if available
        drift_mask, lambda_mat, lambda_mask = self._build_masks_from_causal_spec(
            latent_names, manifest_cols, n_latent, n_manifest
        )

        return SSMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=lambda_mat,
            drift="free",
            diffusion="diag",
            cint="free",  # Enable CINT for non-zero asymptotic means
            manifest_means=None,  # Will be zeros
            manifest_var="diag",
            manifest_dist=manifest_dist,
            manifest_dists=manifest_dists,
            t0_means="free",
            t0_var="diag",
            latent_names=latent_names,
            manifest_names=manifest_cols,
            drift_mask=drift_mask,
            lambda_mask=lambda_mask,
            time_invariant_mask=time_invariant_mask,
        )

    def _build_masks_from_causal_spec(
        self,
        latent_names: list[str] | None,
        manifest_cols: list[str],
        n_latent: int,
        n_manifest: int,
    ) -> tuple[np.ndarray | None, jnp.ndarray, np.ndarray | None]:
        """Build drift_mask and lambda_mask from CausalSpec.

        Args:
            latent_names: Latent construct names (from AR params)
            manifest_cols: Manifest column names (from likelihoods)
            n_latent: Number of latent variables
            n_manifest: Number of manifest variables

        Returns:
            (drift_mask, lambda_mat, lambda_mask) — masks are None when
            causal_spec is not available (backward-compatible).
        """
        if self._causal_spec is None or latent_names is None:
            return None, jnp.eye(n_manifest, n_latent), None

        causal_spec = self._causal_spec
        latent_data = causal_spec.get("latent", {})
        measurement_data = causal_spec.get("measurement", {})
        edges = latent_data.get("edges", [])
        indicators = measurement_data.get("indicators", [])

        # Build name-to-index maps
        latent_idx = {name: i for i, name in enumerate(latent_names)}

        # --- Drift mask ---
        # Diagonal always True (AR effects); off-diagonal True only where
        # a CausalEdge exists between two constructs.
        drift_mask = np.eye(n_latent, dtype=bool)
        for edge in edges:
            cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
            effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
            if cause in latent_idx and effect in latent_idx:
                # drift[effect_idx, cause_idx] = True (effect row, cause col)
                drift_mask[latent_idx[effect], latent_idx[cause]] = True

        # --- Lambda mask ---
        # Build from measurement model indicators → construct mapping.
        # First indicator per construct: fixed at 1.0 (reference indicator).
        # Additional indicators: free to sample (lambda_mask True).
        manifest_idx = {name: i for i, name in enumerate(manifest_cols)}
        lambda_mat_np = np.zeros((n_manifest, n_latent), dtype=np.float64)
        lambda_mask = np.zeros((n_manifest, n_latent), dtype=bool)

        # Track which constructs already have a reference indicator
        reference_set: set[str] = set()

        for indicator in indicators:
            ind_name = indicator.get("name") if isinstance(indicator, dict) else indicator.name
            construct = (
                indicator.get("construct_name")
                if isinstance(indicator, dict)
                else indicator.construct_name
            )

            if ind_name not in manifest_idx or construct not in latent_idx:
                continue

            mi = manifest_idx[ind_name]
            li = latent_idx[construct]

            if construct not in reference_set:
                # First indicator for this construct: fixed reference
                lambda_mat_np[mi, li] = 1.0
                reference_set.add(construct)
            else:
                # Additional indicator: free to sample
                lambda_mask[mi, li] = True

        lambda_mat = jnp.array(lambda_mat_np)

        # If no measurement model indicators matched, fall back to identity
        if not reference_set:
            return drift_mask, jnp.eye(n_manifest, n_latent), None

        return drift_mask, lambda_mat, lambda_mask

    def _convert_priors_to_ssm(
        self,
        priors: dict[str, dict],
        model_spec: ModelSpec | dict | None,
        ssm_spec: SSMSpec | None = None,
    ) -> SSMPriors:
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

        Returns:
            SSMPriors for the model
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
        offdiag_param_index, lambda_param_index = self._build_prior_index_maps(ssm_spec, model_spec)

        for param_name, prior_spec in priors.items():
            distribution = prior_spec.get("distribution", "Normal")
            params = prior_spec.get("params", {})

            # Normalize distribution params to mu/sigma
            normalized = _normalize_prior_params(distribution, params)

            # Check if this parameter maps to a specific array position
            if param_name in offdiag_param_index:
                attr, idx = offdiag_param_index[param_name]
                per_element.setdefault(attr, []).append((idx, normalized))
                continue
            if param_name in lambda_param_index:
                attr, idx = lambda_param_index[param_name]
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
            n_total = max(idx for idx, _ in entries) + 1

            # Build arrays from defaults + positioned entries
            mu_default = current.get("mu", 0.0)
            sigma_default = current.get("sigma", 0.5)

            mu_arr = [float(mu_default)] * n_total
            sigma_arr = [float(sigma_default)] * n_total

            for idx, normed in entries:
                if "mu" in normed:
                    mu_arr[idx] = float(normed["mu"])
                if "sigma" in normed:
                    sigma_arr[idx] = float(normed["sigma"])

            result = {"mu": mu_arr, "sigma": sigma_arr}

            # Propagate bounds if any entry has them
            has_bounds = any("lower" in n for _, n in entries)
            if has_bounds:
                lower_arr = [float(normed.get("lower", -1e6)) for _, normed in entries]
                upper_arr = [float(normed.get("upper", 1e6)) for _, normed in entries]
                result["lower"] = lower_arr
                result["upper"] = upper_arr

            setattr(ssm_priors, attr, result)

        return ssm_priors

    def _build_prior_index_maps(
        self,
        ssm_spec: SSMSpec | None,
        model_spec: ModelSpec | dict | None,
    ) -> tuple[dict[str, tuple[str, int]], dict[str, tuple[str, int]]]:
        """Build parameter name → (SSMPriors field, array index) maps.

        Uses drift_mask and lambda_mask to determine which array position
        each causal parameter occupies, so per-element priors align with
        the sampling order in _sample_drift/_sample_lambda.

        Returns:
            (offdiag_param_index, lambda_param_index) — both are
            {param_name: (ssm_field, index)} dicts. Empty if no masks.
        """
        offdiag_index: dict[str, tuple[str, int]] = {}
        lambda_index: dict[str, tuple[str, int]] = {}

        if ssm_spec is None:
            return offdiag_index, lambda_index

        # Parse model_spec for parameter names + roles
        if not model_spec:
            return offdiag_index, lambda_index
        if isinstance(model_spec, dict):
            spec_obj = ModelSpec.model_validate(model_spec)
        elif isinstance(model_spec, ModelSpec):
            spec_obj = model_spec
        else:
            return offdiag_index, lambda_index

        latent_names = ssm_spec.latent_names or []
        latent_idx_map = {name: i for i, name in enumerate(latent_names)}

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
                    logger.warning(
                        "Could not parse FIXED_EFFECT parameter '%s' into "
                        "(cause, effect) from known latents %s",
                        p.name,
                        sorted(latent_name_set),
                    )
                    continue
                cause_name, effect_name = result
                pos = (latent_idx_map[effect_name], latent_idx_map[cause_name])
                if pos in positions:
                    offdiag_index[p.name] = ("drift_offdiag", positions.index(pos))

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
                    logger.warning(
                        "Could not parse LOADING parameter '%s' into "
                        "(indicator, construct) from known manifests %s / latents %s",
                        p.name,
                        sorted(manifest_name_set),
                        sorted(latent_name_set),
                    )
                    continue
                ind_name, construct_name = result
                pos = (manifest_idx_map[ind_name], latent_idx_map[construct_name])
                if pos in positions:
                    lambda_index[p.name] = ("lambda_free", positions.index(pos))

        return offdiag_index, lambda_index

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
        if self._ssm_spec is not None:
            spec = self._ssm_spec
        elif self._model_spec is not None:
            spec = self._convert_spec_to_ssm(self._model_spec)
        else:
            # Auto-detect from data
            manifest_cols = [
                c for c in X.columns if c not in ["time", "time_bucket"] and not c.endswith("_lag1")
            ]
            spec = SSMSpec(
                n_latent=len(manifest_cols),
                n_manifest=len(manifest_cols),
                lambda_mat=jnp.eye(len(manifest_cols)),
            )

        # Convert priors (pass ssm_spec for per-element positioning)
        priors = self._convert_priors_to_ssm(self._priors, self._model_spec or {}, ssm_spec=spec)

        # Create model with PF config from model_config
        n_particles = self._model_config.get("n_particles", 200)
        pf_seed = self._model_config.get("pf_seed", 0)
        self._model = SSMModel(spec, priors, n_particles=n_particles, pf_seed=pf_seed)
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

        # Prepare data
        observations, times = self._prepare_data(X)

        # Merge sampler config with kwargs
        sampler_config = {**self._sampler_config, **kwargs}

        # Extract method (default to nuts_da) without mutating
        method = sampler_config.get("method", "nuts_da")
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
        if hasattr(self, "_spec") and self._spec.manifest_names:
            manifest_cols = self._spec.manifest_names
        else:
            manifest_cols = [
                c for c in X.columns if c not in ["time", "time_bucket"] and not c.endswith("_lag1")
            ]

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
        from causal_ssm_agent.models.ssm.inference import prior_predictive

        if self._model is None:
            raise ValueError("Model must be built before sampling prior predictive")

        if times is None:
            times = jnp.arange(10, dtype=jnp.float32)
        return prior_predictive(self._model, times, num_samples=samples)

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
