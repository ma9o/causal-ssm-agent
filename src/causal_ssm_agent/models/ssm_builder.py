"""SSM Model Builder for causal SSM pipeline integration.

Provides a model builder interface compatible with the causal SSM pipeline
while using the NumPyro SSM implementation underneath.
"""

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from causal_ssm_agent.models.ssm import (
    InferenceResult,
    NoiseFamily,
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

# Mapping from user-facing DistributionFamily to internal NoiseFamily
_DIST_TO_NOISE: dict[DistributionFamily, NoiseFamily] = {
    DistributionFamily.NORMAL: NoiseFamily.GAUSSIAN,
    DistributionFamily.POISSON: NoiseFamily.POISSON,
    DistributionFamily.GAMMA: NoiseFamily.GAMMA,
    DistributionFamily.NEGATIVE_BINOMIAL: NoiseFamily.POISSON,  # count data → particle
    DistributionFamily.BERNOULLI: NoiseFamily.POISSON,  # discrete → particle
    DistributionFamily.BETA: NoiseFamily.GAUSSIAN,  # fallback
    DistributionFamily.ORDERED_LOGISTIC: NoiseFamily.GAUSSIAN,  # fallback
    DistributionFamily.CATEGORICAL: NoiseFamily.GAUSSIAN,  # fallback
}


# Map ParameterRole to SSMPriors field and default mu/sigma params.
# This replaces the old keyword-matching _PRIOR_RULES.
_ROLE_TO_SSM: dict[ParameterRole, tuple[str, dict]] = {
    ParameterRole.AR_COEFFICIENT: ("drift_diag", {"mu": -0.5, "sigma": 1.0}),
    ParameterRole.FIXED_EFFECT: ("drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    ParameterRole.RESIDUAL_SD: ("diffusion_diag", {"sigma": 1.0}),
    ParameterRole.LOADING: ("lambda_free", {"mu": 0.5, "sigma": 0.5}),
    ParameterRole.RANDOM_INTERCEPT_SD: ("pop_sd", {"sigma": 1.0}),
    ParameterRole.RANDOM_SLOPE_SD: ("pop_sd", {"sigma": 1.0}),
    ParameterRole.CORRELATION: ("drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
}

# Fallback keyword matching for parameters without a role in the ModelSpec
# (e.g. when priors are provided as a flat dict without ParameterSpec context)
_KEYWORD_RULES: list[tuple[list[str], str, dict]] = [
    (["rho", "ar"], "drift_diag", {"mu": -0.5, "sigma": 1.0}),
    (["beta"], "drift_offdiag", {"mu": 0.0, "sigma": 0.5}),
    (["sigma", "sd"], "diffusion_diag", {"sigma": 1.0}),
    (["lambda", "loading"], "lambda_free", {"mu": 0.5, "sigma": 0.5}),
    (["tau"], "pop_sd", {"sigma": 1.0}),
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
        # Convert Uniform(lower, upper) to Normal(midpoint, range/4)
        # range/4 ~ 95% interval
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 4
        return {"mu": mu, "sigma": sigma}

    # Fallback: try to extract mu/sigma directly
    return {"mu": params.get("mu", 0.0), "sigma": params.get("sigma", 1.0)}


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
    ):
        """Initialize the SSM model builder.

        Args:
            model_spec: Model specification from orchestrator (will be converted)
            priors: Prior proposals for each parameter
            ssm_spec: Direct SSMSpec (overrides model_spec conversion)
            model_config: Override model configuration (n_particles, pf_seed)
            sampler_config: Override sampler configuration
        """
        self._model_spec = model_spec
        self._priors = priors or {}
        self._ssm_spec = ssm_spec
        self._model_config = model_config or {}
        self._sampler_config = sampler_config or self.get_default_sampler_config()

        self._model: SSMModel | None = None
        self._result: InferenceResult | None = None

    @staticmethod
    def get_default_sampler_config() -> dict:
        """Default sampler configuration."""
        return {
            "num_warmup": 1000,
            "num_samples": 1000,
            "num_chains": 4,
            "seed": 0,
        }

    def _convert_spec_to_ssm(self, model_spec: ModelSpec | dict, data: pd.DataFrame) -> SSMSpec:
        """Convert ModelSpec to SSMSpec.

        This is a heuristic conversion that maps the ModelSpec
        to continuous-time parameters.

        Args:
            model_spec: Model specification
            data: Data frame with indicator columns

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

        # Check for hierarchical structure
        hierarchical = len(model_spec.random_effects) > 0
        n_subjects = 1
        if hierarchical and "subject_id" in data.columns:
            n_subjects = data["subject_id"].nunique()

        # Determine manifest noise family from likelihoods
        manifest_dist = NoiseFamily.GAUSSIAN
        for lik in model_spec.likelihoods:
            noise = _DIST_TO_NOISE.get(lik.distribution, NoiseFamily.GAUSSIAN)
            if noise != NoiseFamily.GAUSSIAN:
                manifest_dist = noise
                break  # Any non-Gaussian triggers particle filter

        # Derive latent names from AR parameter names (e.g. rho_X → X)
        latent_names = [p.name.removeprefix("rho_") for p in ar_params] if ar_params else None

        # Create default lambda matrix (identity mapping)
        lambda_mat = jnp.eye(n_manifest, n_latent)

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
            t0_means="free",
            t0_var="diag",
            hierarchical=hierarchical,
            n_subjects=n_subjects,
            indvarying=["t0_means"],  # Allow individual variation in initial states
            latent_names=latent_names,
            manifest_names=manifest_cols,
        )

    def _convert_priors_to_ssm(
        self, priors: dict[str, dict], model_spec: ModelSpec | dict | None
    ) -> SSMPriors:
        """Convert prior proposals to SSMPriors.

        Uses ParameterRole from ModelSpec to determine which SSMPriors field
        each prior maps to, then normalizes distribution-specific params
        (Beta alpha/beta, Uniform lower/upper) to the mu/sigma format
        that SSMPriors expects.

        Falls back to keyword matching when ModelSpec is not available.

        Args:
            priors: Prior proposals from workers
            model_spec: Model specification for context (optional)

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

        for param_name, prior_spec in priors.items():
            distribution = prior_spec.get("distribution", "Normal")
            params = prior_spec.get("params", {})

            # Normalize distribution params to mu/sigma
            normalized = _normalize_prior_params(distribution, params)

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
                for keywords, attr, defaults in _KEYWORD_RULES:
                    if any(kw in name_lower for kw in keywords):
                        merged = {k: normalized.get(k, v) for k, v in defaults.items()}
                        setattr(ssm_priors, attr, merged)
                        break

        return ssm_priors

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> SSMModel:
        """Build the NumPyro SSM model.

        Args:
            X: Data with indicator columns, time, and optional subject_id
            y: Optional target (if not in X)

        Returns:
            The constructed SSMModel
        """
        # Determine specification
        if self._ssm_spec is not None:
            spec = self._ssm_spec
        elif self._model_spec is not None:
            spec = self._convert_spec_to_ssm(self._model_spec, X)
        else:
            # Auto-detect from data
            manifest_cols = [
                c
                for c in X.columns
                if c not in ["time", "time_bucket", "subject_id", "subject"]
                and not c.endswith("_lag1")
            ]
            spec = SSMSpec(
                n_latent=len(manifest_cols),
                n_manifest=len(manifest_cols),
                lambda_mat=jnp.eye(len(manifest_cols)),
                hierarchical="subject_id" in X.columns or "subject" in X.columns,
                n_subjects=X.get("subject_id", X.get("subject", pd.Series([0]))).nunique(),
            )

        # Convert priors
        priors = self._convert_priors_to_ssm(self._priors, self._model_spec or {})

        # Create model with PF config from model_config
        n_particles = self._model_config.get("n_particles", 200)
        pf_seed = self._model_config.get("pf_seed", 0)
        self._model = SSMModel(spec, priors, n_particles=n_particles, pf_seed=pf_seed)
        self._spec = spec

        return self._model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        """Fit the SSM model to data.

        Args:
            X: Data with indicator columns, time, and optional subject_id
            y: Optional target (if not in X)
            **kwargs: Additional arguments passed to inference

        Returns:
            InferenceResult with posterior samples
        """
        if self._model is None:
            self.build_model(X, y)

        # Prepare data
        observations, times, subject_ids = self._prepare_data(X)

        # Merge sampler config with kwargs
        sampler_config = {**self._sampler_config, **kwargs}

        # Extract method (default to svi)
        method = sampler_config.pop("method", "svi")

        result = fit(
            self._model,
            observations=observations,
            times=times,
            subject_ids=subject_ids,
            method=method,
            **sampler_config,
        )

        self._result = result
        return result

    def _prepare_data(self, X: pd.DataFrame) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Prepare data for SSM fitting.

        Args:
            X: DataFrame with observations

        Returns:
            Tuple of (observations, times, subject_ids)
        """
        # Get manifest columns
        if hasattr(self, "_spec") and self._spec.manifest_names:
            manifest_cols = self._spec.manifest_names
        else:
            manifest_cols = [
                c
                for c in X.columns
                if c not in ["time", "time_bucket", "subject_id", "subject"]
                and not c.endswith("_lag1")
            ]

        # Extract observations
        observations = jnp.array(X[manifest_cols].values, dtype=jnp.float32)

        # Extract times
        time_col = "time" if "time" in X.columns else "time_bucket"
        if time_col in X.columns:
            times = jnp.array(X[time_col].values, dtype=jnp.float32)
        else:
            # Default: integer sequence
            times = jnp.arange(len(X), dtype=jnp.float32)

        # Extract subject IDs
        subject_col = "subject_id" if "subject_id" in X.columns else "subject"
        if subject_col in X.columns:
            # Convert to 0-indexed integers
            subject_ids = jnp.array(pd.factorize(X[subject_col])[0], dtype=jnp.int32)
        else:
            subject_ids = None

        return observations, times, subject_ids

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

    def summary(self) -> pd.DataFrame:
        """Get summary statistics for posterior.

        Returns:
            DataFrame with summary statistics
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
        return pd.DataFrame(summary_data)
