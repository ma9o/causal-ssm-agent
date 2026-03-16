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
from causal_ssm_agent.models.ssm_compilation import (
    bind_parameters as bind_parameters_core,
)
from causal_ssm_agent.models.ssm_compilation import (
    build_prior_index_maps as build_prior_index_maps_core,
)
from causal_ssm_agent.models.ssm_compilation import (
    check_drift_lag_consistency as check_drift_lag_consistency_core,
)
from causal_ssm_agent.models.ssm_compilation import (
    compile_priors as compile_priors_core,
)
from causal_ssm_agent.models.ssm_compilation import (
    compile_ssm_inputs as compile_ssm_inputs_core,
)
from causal_ssm_agent.models.ssm_compilation import (
    translate_spec as translate_spec_core,
)
from causal_ssm_agent.models.ssm_compilation import (
    warn_first_order_approximation as warn_first_order_approximation_core,
)
from causal_ssm_agent.models.ssm_observation_metadata import (
    default_manifest_columns,
    hydrate_discrete_manifest_metadata,
    validate_observation_support,
)
from causal_ssm_agent.orchestrator.schemas_model import (
    DistributionFamily,
    ModelSpec,
)
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

    def compile_inputs(self) -> tuple[SSMSpec, SSMPriors]:
        """Compile user-facing specs into executable SSM inputs."""
        if self._ssm_spec is not None and self._ssm_priors is not None:
            if self._parameter_bindings is None and self._model_spec is not None:
                self._parameter_bindings = bind_parameters_core(
                    self._model_spec,
                    self._ssm_spec,
                    causal_spec=self._causal_spec,
                )
            return self._ssm_spec, self._ssm_priors

        if self._ssm_spec is None and self._model_spec is not None and self._ssm_priors is None:
            raw_priors: dict[str, dict] = {
                k: v.model_dump() if isinstance(v, PriorProposal) else v
                for k, v in self._priors.items()
            }
            spec, priors, bindings = compile_ssm_inputs_core(
                self._model_spec,
                raw_priors,
                causal_spec=self._causal_spec,
            )
            if self._parameter_bindings is None:
                self._parameter_bindings = bindings
            return spec, priors

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

    def _convert_spec_to_ssm(
        self, model_spec: ModelSpec | dict
    ) -> tuple[SSMSpec, dict[tuple[int, int], float]]:
        """Compatibility wrapper around the pure spec-translation stage."""
        return translate_spec_core(model_spec, causal_spec=self._causal_spec)

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
        """Compatibility wrapper around the pure prior-compilation stage."""
        return compile_priors_core(
            priors,
            model_spec or {},
            ssm_spec,
            edge_lag_days=edge_lag_days,
            causal_spec=self._causal_spec,
        )

    @staticmethod
    def _warn_first_order_approximation(ssm_priors: SSMPriors) -> None:
        warn_first_order_approximation_core(ssm_priors)

    def _check_drift_lag_consistency(
        self,
        ssm_priors: SSMPriors,
        ssm_spec: SSMSpec,
        edge_lag_days: dict[tuple[int, int], float] | None = None,
    ) -> None:
        check_drift_lag_consistency_core(
            ssm_priors,
            ssm_spec,
            edge_lag_days=edge_lag_days,
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
        """Compatibility wrapper around the pure prior-index stage."""
        return build_prior_index_maps_core(
            ssm_spec,
            model_spec,
            causal_spec=self._causal_spec,
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
        """Compatibility wrapper around the pure binding stage."""
        if model_spec is None:
            return []
        return bind_parameters_core(
            model_spec,
            ssm_spec,
            index_maps=index_maps,
            causal_spec=self._causal_spec,
        )

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
        return hydrate_discrete_manifest_metadata(spec, X)

    def _validate_observation_support(self, spec: SSMSpec, X: Any) -> None:
        """Reject likelihoods whose support is incompatible with observed data."""
        validate_observation_support(spec, X)

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
            manifest_cols = default_manifest_columns(X)

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
    """Pure function: convert ModelSpec to SSMSpec with edge lags."""
    return translate_spec_core(model_spec, causal_spec=causal_spec)


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
    """Pure function: convert prior proposals to SSMPriors."""
    return compile_priors_core(
        raw_priors,
        model_spec,
        ssm_spec,
        edge_lag_days=edge_lag_days,
        causal_spec=causal_spec,
    )


def build_prior_index_maps(
    ssm_spec: SSMSpec,
    model_spec: ModelSpec | dict | None,
) -> tuple[
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
    dict[str, tuple[str, int]],
]:
    """Pure function: build shared index maps for priors and parameter bindings."""
    return build_prior_index_maps_core(ssm_spec, model_spec)


def bind_parameters(
    model_spec: ModelSpec | dict,
    ssm_spec: SSMSpec,
    index_maps: tuple[
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
        dict[str, tuple[str, int]],
    ]
    | None = None,
) -> list[dict[str, Any]]:
    """Pure function: map semantic parameter names to NumPyro sample sites."""
    resolved_index_maps = index_maps or build_prior_index_maps_core(ssm_spec, model_spec)
    return bind_parameters_core(
        model_spec,
        ssm_spec,
        index_maps=resolved_index_maps,
    )


def compile_ssm_inputs(
    model_spec: ModelSpec | dict,
    priors: dict[str, dict],
    causal_spec: dict | None = None,
) -> tuple[SSMSpec, SSMPriors, list[dict[str, Any]]]:
    """Pure function: full compilation pipeline as function composition."""
    return compile_ssm_inputs_core(model_spec, priors, causal_spec=causal_spec)


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
