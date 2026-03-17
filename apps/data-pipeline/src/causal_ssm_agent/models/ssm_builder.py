"""SSM Model Builder for causal SSM pipeline integration."""

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
from causal_ssm_agent.models.ssm_compilation import compile_ssm_inputs
from causal_ssm_agent.models.ssm_compilation_common import dump_prior_payloads
from causal_ssm_agent.models.ssm_observation_metadata import (
    default_manifest_columns,
    hydrate_discrete_manifest_metadata,
    validate_observation_support,
)
from causal_ssm_agent.orchestrator.schemas_model import ModelSpec
from causal_ssm_agent.workers.schemas_prior import PriorProposal


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
                model indicators. When provided, spec translation builds
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

    def compile_inputs(self) -> tuple[SSMSpec, SSMPriors]:
        """Compile user-facing specs into executable SSM inputs."""
        spec, priors, bindings = compile_ssm_inputs(
            model_spec=self._model_spec,
            priors=dump_prior_payloads(self._priors),
            ssm_spec=self._ssm_spec,
            ssm_priors=self._ssm_priors,
            causal_spec=self._causal_spec,
        )
        if self._parameter_bindings is None:
            self._parameter_bindings = bindings
        return spec, priors

    @staticmethod
    def get_default_sampler_config() -> dict:
        """Default sampler configuration, read from config.yaml."""
        from causal_ssm_agent.utils.config import get_config

        return get_config().inference.to_sampler_config()

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
            manifest_cols = default_manifest_columns(X)
            spec = SSMSpec(
                n_latent=len(manifest_cols),
                n_manifest=len(manifest_cols),
                lambda_mat=jnp.eye(len(manifest_cols)),
            )
            spec, priors, _bindings = compile_ssm_inputs(
                priors=dump_prior_payloads(self._priors),
                ssm_spec=spec,
                causal_spec=self._causal_spec,
            )
        else:
            spec, priors = self.compile_inputs()

        spec = hydrate_discrete_manifest_metadata(spec, X)
        validate_observation_support(spec, X)

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

        observations, times, _manifest_names = self.prepare_fit_inputs(X)
        return self.fit_prepared(observations, times, **kwargs)

    def fit_prepared(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        **kwargs: Any,
    ) -> InferenceResult:
        """Fit the built model from precomputed observation/time arrays."""
        if self._model is None:
            raise ValueError("Model must be built before fitting prepared inputs")

        sampler_config = {**self._sampler_config, **kwargs}
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

    def prepare_fit_inputs(
        self,
        X: pl.DataFrame,
    ) -> tuple[jnp.ndarray, jnp.ndarray, list[str]]:
        """Extract arrays and manifest ordering from wide-format fit data.

        Expects wide-format data from pivot_to_wide() which already converts
        datetimes to fractional days. The 'time' column should be numeric.

        Args:
            X: Polars DataFrame with observations (wide format)

        Returns:
            Tuple of (observations, times, manifest_names)
        """
        if self._spec is not None and self._spec.manifest_names:
            manifest_cols = self._spec.manifest_names
        else:
            manifest_cols = default_manifest_columns(X)

        observations = jnp.array(X.select(manifest_cols).to_numpy(), dtype=jnp.float32)

        time_col = "time" if "time" in X.columns else "time_bucket"
        if time_col in X.columns:
            times = jnp.array(X[time_col].to_numpy(), dtype=jnp.float32)
        else:
            times = jnp.arange(X.height, dtype=jnp.float32)

        return observations, times, manifest_cols

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

        from causal_ssm_agent.models.likelihoods.observation_families import (
            any_family_needs_level_metadata,
        )
        from causal_ssm_agent.models.ssm.prior_predictive_runtime import (
            sample_prior_predictive_from_compiled_semantics,
            sample_prior_predictive_from_priors,
        )

        manifest_dists = spec.manifest_dists or [spec.manifest_dist] * spec.n_manifest
        needs_hydration = any_family_needs_level_metadata(manifest_dists)
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


def build_ssm_builder(
    wide_data: pl.DataFrame,
    model_spec: ModelSpec | dict | None = None,
    priors: dict[str, PriorProposal] | dict[str, dict] | None = None,
    causal_spec: dict | None = None,
    sampler_config: dict | None = None,
    compiled_ssm: dict | None = None,
) -> SSMModelBuilder:
    """Single canonical entry point for constructing a ready-to-use SSMModelBuilder.

    The caller owns raw->wide conversion. This function accepts only wide data,
    builds the model once, and returns a ready-to-fit builder.

    Args:
        model_spec: Model specification (dict or ModelSpec)
        priors: Prior proposals by parameter name
        wide_data: Pivoted model matrix with manifest columns and time
        causal_spec: CausalSpec dict for DAG-constrained masks
        sampler_config: Override sampler configuration
        compiled_ssm: Serialized compiled artifact from stage 4

    Returns:
        A fully built SSMModelBuilder (model constructed, ready for fit/sample)

    Raises:
        ValueError: If wide_data is empty
    """
    if compiled_ssm is not None:
        from causal_ssm_agent.models.ssm_compiler import build_compiled_ssm_builder

        return build_compiled_ssm_builder(
            compiled_ssm,
            wide_data,
            sampler_config=sampler_config,
        )

    if wide_data.is_empty():
        raise ValueError("Cannot build SSM model from empty data")

    builder = SSMModelBuilder(
        model_spec=model_spec,
        priors=priors,
        causal_spec=causal_spec,
        sampler_config=sampler_config,
    )
    builder.build_model(wide_data)
    return builder
