"""CT-SEM Model Builder for DSEM pipeline integration.

Provides a model builder interface compatible with the DSEM pipeline
while using the NumPyro CT-SEM implementation underneath.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.infer import MCMC

from dsem_agent.models.ctsem import CTSEMModel, CTSEMPriors, CTSEMSpec
from dsem_agent.orchestrator.schemas_glmm import GLMMSpec, ParameterRole
from dsem_agent.workers.schemas_prior import PriorProposal


class CTSEMModelBuilder:
    """Model builder for CT-SEM using NumPyro.

    This class provides an interface compatible with the DSEM pipeline,
    translating from the GLMMSpec to CTSEMSpec internally.
    """

    _model_type = "CT-SEM"
    version = "0.1.0"

    def __init__(
        self,
        glmm_spec: GLMMSpec | dict | None = None,
        priors: dict[str, PriorProposal] | dict[str, dict] | None = None,
        ctsem_spec: CTSEMSpec | None = None,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        """Initialize the CT-SEM model builder.

        Args:
            glmm_spec: GLMM specification from orchestrator (will be converted)
            priors: Prior proposals for each parameter
            ctsem_spec: Direct CTSEMSpec (overrides glmm_spec conversion)
            model_config: Override model configuration
            sampler_config: Override sampler configuration
        """
        self._glmm_spec = glmm_spec
        self._priors = priors or {}
        self._ctsem_spec = ctsem_spec
        self._model_config = model_config or {}
        self._sampler_config = sampler_config or self.get_default_sampler_config()

        self._model: CTSEMModel | None = None
        self._mcmc: MCMC | None = None

    @staticmethod
    def get_default_sampler_config() -> dict:
        """Default sampler configuration."""
        return {
            "num_warmup": 1000,
            "num_samples": 1000,
            "num_chains": 4,
            "seed": 0,
        }

    def _convert_glmm_to_ctsem(
        self, glmm_spec: GLMMSpec | dict, data: pd.DataFrame
    ) -> CTSEMSpec:
        """Convert GLMMSpec to CTSEMSpec.

        This is a heuristic conversion that maps the discrete-time
        GLMM specification to continuous-time parameters.

        Args:
            glmm_spec: GLMM specification
            data: Data frame with indicator columns

        Returns:
            CTSEMSpec for continuous-time model
        """
        if isinstance(glmm_spec, dict):
            from dsem_agent.orchestrator.schemas_glmm import GLMMSpec
            glmm_spec = GLMMSpec.model_validate(glmm_spec)

        # Extract dimensions from data
        manifest_cols = [lik.variable for lik in glmm_spec.likelihoods]
        n_manifest = len(manifest_cols)

        # Infer latent structure from parameters
        # Look for AR coefficients to determine number of latent processes
        ar_params = [
            p for p in glmm_spec.parameters if p.role == ParameterRole.AR_COEFFICIENT
        ]
        n_latent = max(len(ar_params), 1)

        # Check for hierarchical structure
        hierarchical = len(glmm_spec.random_effects) > 0
        n_subjects = 1
        if hierarchical and "subject_id" in data.columns:
            n_subjects = data["subject_id"].nunique()

        # Create default lambda matrix (identity mapping)
        lambda_mat = jnp.eye(n_manifest, n_latent)

        return CTSEMSpec(
            n_latent=n_latent,
            n_manifest=n_manifest,
            lambda_mat=lambda_mat,
            drift="free",
            diffusion="diag",
            cint="free",  # Enable CINT for non-zero asymptotic means
            manifest_means=None,  # Will be zeros
            manifest_var="diag",
            t0_means="free",
            t0_var="diag",
            hierarchical=hierarchical,
            n_subjects=n_subjects,
            indvarying=["t0_means"],  # Allow individual variation in initial states
            manifest_names=manifest_cols,
        )

    def _convert_priors_to_ctsem(
        self, priors: dict[str, dict], glmm_spec: GLMMSpec | dict | None
    ) -> CTSEMPriors:
        """Convert prior proposals to CTSEMPriors.

        Maps the GLMM parameter priors to the CT-SEM parameterization.

        Args:
            priors: Prior proposals from workers
            glmm_spec: GLMM specification for context (optional)

        Returns:
            CTSEMPriors for the model
        """
        ctsem_priors = CTSEMPriors()

        # Skip GLMMSpec validation if empty or None - just use priors directly
        if glmm_spec and isinstance(glmm_spec, dict) and glmm_spec.get("likelihoods"):
            from dsem_agent.orchestrator.schemas_glmm import GLMMSpec
            glmm_spec = GLMMSpec.model_validate(glmm_spec)

        # Map AR coefficients to drift diagonal
        # In CT-SEM, drift diagonal ≈ log(AR coefficient) / dt
        # For simplicity, use the prior for AR as a guide for drift
        for param_name, prior_spec in priors.items():
            if "rho" in param_name.lower() or "ar" in param_name.lower():
                # AR coefficient prior -> drift diagonal prior
                # Typical AR(1) in [0, 1] -> drift in [-inf, 0]
                mu = prior_spec.get("params", {}).get("mu", -0.5)
                sigma = prior_spec.get("params", {}).get("sigma", 1.0)
                ctsem_priors.drift_diag = {"mu": mu, "sigma": sigma}

            elif "beta" in param_name.lower():
                # Cross-lag coefficient -> drift off-diagonal
                mu = prior_spec.get("params", {}).get("mu", 0.0)
                sigma = prior_spec.get("params", {}).get("sigma", 0.5)
                ctsem_priors.drift_offdiag = {"mu": mu, "sigma": sigma}

            elif "sigma" in param_name.lower() or "sd" in param_name.lower():
                # Residual SD -> diffusion diagonal
                sigma = prior_spec.get("params", {}).get("sigma", 1.0)
                ctsem_priors.diffusion_diag = {"sigma": sigma}

        return ctsem_priors

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> CTSEMModel:
        """Build the NumPyro CT-SEM model.

        Args:
            X: Data with indicator columns, time, and optional subject_id
            y: Optional target (if not in X)

        Returns:
            The constructed CTSEMModel
        """
        # Determine specification
        if self._ctsem_spec is not None:
            spec = self._ctsem_spec
        elif self._glmm_spec is not None:
            spec = self._convert_glmm_to_ctsem(self._glmm_spec, X)
        else:
            # Auto-detect from data
            manifest_cols = [
                c for c in X.columns
                if c not in ["time", "time_bucket", "subject_id", "subject"]
                and not c.endswith("_lag1")
            ]
            spec = CTSEMSpec(
                n_latent=len(manifest_cols),
                n_manifest=len(manifest_cols),
                lambda_mat=jnp.eye(len(manifest_cols)),
                hierarchical="subject_id" in X.columns or "subject" in X.columns,
                n_subjects=X.get("subject_id", X.get("subject", pd.Series([0]))).nunique(),
            )

        # Convert priors
        priors = self._convert_priors_to_ctsem(self._priors, self._glmm_spec or {})

        # Create model
        self._model = CTSEMModel(spec, priors)
        self._spec = spec

        return self._model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> MCMC:
        """Fit the CT-SEM model to data.

        Args:
            X: Data with indicator columns, time, and optional subject_id
            y: Optional target (if not in X)
            **kwargs: Additional arguments passed to MCMC

        Returns:
            MCMC object with posterior samples
        """
        if self._model is None:
            self.build_model(X, y)

        # Prepare data
        observations, times, subject_ids = self._prepare_data(X)

        # Merge sampler config with kwargs
        sampler_config = {**self._sampler_config, **kwargs}

        # Fit
        self._mcmc = self._model.fit(
            observations=observations,
            times=times,
            subject_ids=subject_ids,
            **sampler_config,
        )

        return self._mcmc

    def _prepare_data(
        self, X: pd.DataFrame
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """Prepare data for CT-SEM fitting.

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
                c for c in X.columns
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
            subject_ids = jnp.array(
                pd.factorize(X[subject_col])[0], dtype=jnp.int32
            )
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
        if self._model is None:
            raise ValueError("Model must be built before sampling prior predictive")

        times = jnp.arange(10, dtype=jnp.float32)
        return self._model.prior_predictive(times, num_samples=samples)

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Get posterior samples.

        Returns:
            Dict of posterior samples
        """
        if self._mcmc is None:
            raise ValueError("Model must be fit before getting samples")
        return self._mcmc.get_samples()

    def summary(self) -> pd.DataFrame:
        """Get summary statistics for posterior.

        Returns:
            DataFrame with summary statistics
        """
        if self._mcmc is None:
            raise ValueError("Model must be fit before getting summary")

        self._mcmc.print_summary()

        # Also return as DataFrame
        samples = self.get_samples()
        summary_data = []
        for name, values in samples.items():
            if values.ndim == 1:
                summary_data.append({
                    "parameter": name,
                    "mean": float(jnp.mean(values)),
                    "std": float(jnp.std(values)),
                    "5%": float(jnp.percentile(values, 5)),
                    "95%": float(jnp.percentile(values, 95)),
                })
        return pd.DataFrame(summary_data)


def glmm_to_ctsem_spec(
    glmm_spec: GLMMSpec,
    n_latent: int | None = None,
    n_manifest: int | None = None,
) -> CTSEMSpec:
    """Convert a GLMMSpec to CTSEMSpec.

    This is a standalone conversion function for cases where you have
    a GLMMSpec but want to fit a CT-SEM model.

    Args:
        glmm_spec: GLMM specification
        n_latent: Number of latent processes (inferred if None)
        n_manifest: Number of manifest indicators (inferred if None)

    Returns:
        CTSEMSpec for continuous-time model
    """
    # Infer dimensions
    if n_manifest is None:
        n_manifest = len(glmm_spec.likelihoods)

    if n_latent is None:
        ar_params = [
            p for p in glmm_spec.parameters if p.role == ParameterRole.AR_COEFFICIENT
        ]
        n_latent = max(len(ar_params), n_manifest)

    # Check for hierarchical structure
    hierarchical = len(glmm_spec.random_effects) > 0

    return CTSEMSpec(
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
        manifest_names=[lik.variable for lik in glmm_spec.likelihoods],
    )
