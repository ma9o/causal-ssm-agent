"""SSM Model Builder for DSEM pipeline integration.

Provides a model builder interface compatible with the DSEM pipeline
while using the NumPyro SSM implementation underneath.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from dsem_agent.models.ssm import InferenceResult, NoiseFamily, SSMModel, SSMPriors, SSMSpec, fit
from dsem_agent.orchestrator.schemas_model import DistributionFamily, ModelSpec, ParameterRole
from dsem_agent.workers.schemas_prior import PriorProposal

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


class SSMModelBuilder:
    """Model builder for SSM using NumPyro.

    This class provides an interface compatible with the DSEM pipeline,
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
            from dsem_agent.orchestrator.schemas_model import ModelSpec

            model_spec = ModelSpec.model_validate(model_spec)

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
            manifest_names=manifest_cols,
        )

    def _convert_priors_to_ssm(
        self, priors: dict[str, dict], model_spec: ModelSpec | dict | None
    ) -> SSMPriors:
        """Convert prior proposals to SSMPriors.

        Maps the ModelSpec parameter priors to the SSM parameterization.

        Args:
            priors: Prior proposals from workers
            model_spec: Model specification for context (optional)

        Returns:
            SSMPriors for the model
        """
        ssm_priors = SSMPriors()

        # Skip ModelSpec validation if empty or None - just use priors directly
        if model_spec and isinstance(model_spec, dict) and model_spec.get("likelihoods"):
            from dsem_agent.orchestrator.schemas_model import ModelSpec

            model_spec = ModelSpec.model_validate(model_spec)

        # Map AR coefficients to drift diagonal
        # In SSM, drift diagonal ≈ log(AR coefficient) / dt
        # For simplicity, use the prior for AR as a guide for drift
        for param_name, prior_spec in priors.items():
            if "rho" in param_name.lower() or "ar" in param_name.lower():
                # AR coefficient prior -> drift diagonal prior
                # Typical AR(1) in [0, 1] -> drift in [-inf, 0]
                mu = prior_spec.get("params", {}).get("mu", -0.5)
                sigma = prior_spec.get("params", {}).get("sigma", 1.0)
                ssm_priors.drift_diag = {"mu": mu, "sigma": sigma}

            elif "beta" in param_name.lower():
                # Cross-lag coefficient -> drift off-diagonal
                mu = prior_spec.get("params", {}).get("mu", 0.0)
                sigma = prior_spec.get("params", {}).get("sigma", 0.5)
                ssm_priors.drift_offdiag = {"mu": mu, "sigma": sigma}

            elif "sigma" in param_name.lower() or "sd" in param_name.lower():
                # Residual SD -> diffusion diagonal
                sigma = prior_spec.get("params", {}).get("sigma", 1.0)
                ssm_priors.diffusion_diag = {"sigma": sigma}

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

    def sample_prior_predictive(self, samples: int = 500) -> Any:
        """Sample from the prior predictive distribution.

        Args:
            samples: Number of samples

        Returns:
            Prior predictive samples
        """
        from dsem_agent.models.ssm.inference import prior_predictive

        if self._model is None:
            raise ValueError("Model must be built before sampling prior predictive")

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


def model_spec_to_ssm_spec(
    model_spec: ModelSpec,
    n_latent: int | None = None,
    n_manifest: int | None = None,
) -> SSMSpec:
    """Convert a ModelSpec to SSMSpec.

    This is a standalone conversion function for cases where you have
    a ModelSpec but want to fit a SSM model.

    Args:
        model_spec: Model specification
        n_latent: Number of latent processes (inferred if None)
        n_manifest: Number of manifest indicators (inferred if None)

    Returns:
        SSMSpec for continuous-time model
    """
    # Infer dimensions
    if n_manifest is None:
        n_manifest = len(model_spec.likelihoods)

    if n_latent is None:
        ar_params = [p for p in model_spec.parameters if p.role == ParameterRole.AR_COEFFICIENT]
        n_latent = max(len(ar_params), 1)

    # Check for hierarchical structure
    hierarchical = len(model_spec.random_effects) > 0

    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        lambda_mat=jnp.eye(n_manifest, n_latent),
        drift="free",
        diffusion="diag",
        cint="free",
        manifest_means=None,
        manifest_var="diag",
        t0_means="free",
        t0_var="diag",
        hierarchical=hierarchical,
        indvarying=["t0_means"],
        manifest_names=[lik.variable for lik in model_spec.likelihoods],
    )
