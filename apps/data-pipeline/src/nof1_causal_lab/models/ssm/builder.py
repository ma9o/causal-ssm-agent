"""SSM Model Builder for N-of-1 Causal Lab pipeline integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import polars as pl

from nof1_causal_lab.models.ssm import (
    InferenceResult,
    PriorRegistry,
    SSMModel,
    SSMParameterLayout,
    SSMSpec,
    fit,
    full_cholesky_mask,
    full_diagonal_mask,
    full_drift_offdiag_mask,
    full_vector_mask,
    strict_lower_triangle_mask,
    zero_loading_mask,
    zero_vector_mask,
)
from nof1_causal_lab.models.ssm.compile.common import dump_prior_payloads
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_ssm_inputs_from_model_spec,
    compile_ssm_inputs_from_spec,
)
from nof1_causal_lab.models.ssm.dynamics.composite import linear_drift_spec
from nof1_causal_lab.models.ssm.inference.structure import (
    InferenceStructurePlan,
    plan_inference_structure,
)
from nof1_causal_lab.models.ssm.observation_support import (
    ObservationSupportRuntime,
    augment_wide_data_with_support_boundaries,
    compile_observation_support_runtime,
    default_manifest_columns,
    hydrate_discrete_manifest_metadata,
    validate_observation_support,
)
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    load_prior_runtime_bundle,
)
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from nof1_causal_lab.utils.data import pivot_to_wide

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.workers.schemas_prior import PriorProposal

logger = logging.getLogger(__name__)


def _center_manifest_columns(
    X: pl.DataFrame,
    manifest_cols: list[str],
    manifest_centered: list[bool] | None,
) -> pl.DataFrame:
    """Apply deterministic centering to manifest columns marked centered."""
    if manifest_centered is None or not any(manifest_centered):
        return X

    centered_exprs = []
    for manifest_name, centered in zip(manifest_cols, manifest_centered, strict=False):
        base_expr = pl.col(manifest_name).cast(pl.Float64, strict=False)
        if centered:
            centered_exprs.append((base_expr - base_expr.mean()).alias(manifest_name))
        else:
            centered_exprs.append(base_expr.alias(manifest_name))
    passthrough = [col_name for col_name in X.columns if col_name not in set(manifest_cols)]
    return X.select(*passthrough, *centered_exprs)


@dataclass
class PreparedModelRuntime:
    """Canonical prepared runtime context shared by validation and inference."""

    builder: SSMModelBuilder
    model: SSMModel
    spec: SSMSpec
    parameter_layout: SSMParameterLayout
    wide_data: pl.DataFrame
    observation_data: pl.DataFrame | None
    observation_support: ObservationSupportRuntime | None
    inference_structure: InferenceStructurePlan
    observations: jnp.ndarray  # (T, n_manifest)
    times: jnp.ndarray  # (T,)
    transition_inputs: jnp.ndarray | None  # (T, n_input)
    manifest_names: list[str]


class SSMModelBuilder:
    """Model builder for SSM using NumPyro.

    This class provides an interface compatible with the N-of-1 Causal Lab pipeline,
    translating from the ModelSpec to SSMSpec internally.
    """

    _model_type = "SSM"
    version = "0.1.0"

    def __init__(
        self,
        model_spec: ModelSpec | dict | None = None,
        priors: dict[str, PriorProposal] | dict[str, dict] | None = None,
        ssm_spec: SSMSpec | None = None,
        prior_registry: PriorRegistry | None = None,
        compiled_prior_semantics: dict | None = None,
        prior_runtime_bundle: PriorRuntimeBundle | None = None,
        sampler_config: dict | None = None,
        causal_spec: dict | None = None,
        parameter_bindings: list[dict[str, Any]] | None = None,
    ):
        """Initialize the SSM model builder.

        Args:
            model_spec: Model specification from orchestrator (will be converted)
            priors: Prior proposals for each parameter
            ssm_spec: Direct SSMSpec (overrides model_spec conversion)
            prior_registry: Direct prior registry paired with ssm_spec
            sampler_config: Override sampler configuration
            causal_spec: CausalSpec dict with latent model edges and measurement
                model indicators. When provided, spec translation builds
                drift_mask and lambda_mask from the DAG structure.
        """
        if model_spec is not None and (ssm_spec is not None or prior_registry is not None):
            raise ValueError(
                "SSMModelBuilder accepts either ModelSpec-driven inputs or direct "
                "SSMSpec inputs, not both."
            )
        self._model_spec = model_spec
        self._priors = priors or {}
        self._spec: SSMSpec | None = ssm_spec
        self._prior_registry = prior_registry
        self._compiled_prior_semantics = compiled_prior_semantics
        self._prior_runtime_bundle = prior_runtime_bundle
        self._sampler_config = sampler_config or self.get_default_sampler_config()
        self._causal_spec = causal_spec
        self._parameter_bindings = parameter_bindings

        self._model: SSMModel | None = None
        self._result: InferenceResult | None = None
        self._prepared_times: jnp.ndarray | None = None
        self._prepared_transition_inputs: jnp.ndarray | None = None
        self._prepared_observation_mask: jnp.ndarray | None = None
        self._prepared_observation_support: ObservationSupportRuntime | None = None

    @property
    def has_model(self) -> bool:
        """Return whether this builder has already materialized an SSMModel."""
        return self._model is not None

    @property
    def model_type(self) -> str:
        """Return the builder's public model-type label."""
        return self._model_type

    @property
    def model(self) -> SSMModel:
        """Return the built SSMModel."""
        if self._model is None:
            raise ValueError("SSMModelBuilder has no built SSMModel")
        return self._model

    @property
    def spec(self) -> SSMSpec:
        """Return the compiled SSMSpec associated with this builder."""
        if self._spec is not None:
            return self._spec
        if self._model is not None:
            return self._model.spec
        raise ValueError("SSMModelBuilder has no compiled SSMSpec")

    def attach_runtime_artifacts(
        self,
        model: SSMModel,
        *,
        result: InferenceResult | None = None,
    ) -> None:
        """Attach an already-built model and optional fit result to this builder."""
        self._model = model
        self._spec = model.spec
        self._result = result

    def _load_prior_runtime_bundle(
        self,
        compiled_prior_semantics: dict[str, Any],
    ) -> PriorRuntimeBundle:
        """Load compiled prior semantics once for runtime consumers."""
        if self._prior_runtime_bundle is None:
            self._prior_runtime_bundle = load_prior_runtime_bundle(compiled_prior_semantics)
        return self._prior_runtime_bundle

    def compile_inputs(self) -> tuple[SSMSpec, PriorRegistry]:
        """Compile user-facing specs into executable SSM inputs."""
        if self._model_spec is not None:
            spec, priors, bindings, _diagnostics, _edge_lag_days = (
                compile_ssm_inputs_from_model_spec(
                    model_spec=self._model_spec,
                    priors=dump_prior_payloads(self._priors),
                    causal_spec=self._causal_spec,
                )
            )
        elif self._spec is not None:
            spec, priors, bindings, _diagnostics, _edge_lag_days = compile_ssm_inputs_from_spec(
                ssm_spec=self._spec,
                priors=dump_prior_payloads(self._priors),
                prior_registry=self._prior_registry,
                model_spec=self._model_spec,
                causal_spec=self._causal_spec,
            )
        else:
            raise ValueError("compile_inputs() requires either model_spec or ssm_spec")
        if self._parameter_bindings is None:
            self._parameter_bindings = bindings
        self._spec = spec
        return spec, priors

    @staticmethod
    def get_default_sampler_config() -> dict:
        """Default sampler configuration, read from config.yaml."""
        from nof1_causal_lab.utils.config import get_config

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
        if self._model_spec is None and self._spec is not None and self._causal_spec is not None:
            raise ValueError(
                "Do not pass causal_spec alongside a direct SSMSpec. "
                "Compile from ModelSpec + CausalSpec or use a compiled artifact so "
                "the causal structure is encoded explicitly in the spec masks."
            )
        if self._spec is None and self._model_spec is None:
            if self._causal_spec is not None:
                raise ValueError(
                    "Cannot auto-detect an SSMSpec when causal_spec is provided. "
                    "Pass ModelSpec + CausalSpec or a compiled Stage 4 artifact so "
                    "drift/loading masks come from the retained causal structure."
                )
            # Auto-detect from data
            manifest_cols = default_manifest_columns(X)
            n = len(manifest_cols)
            spec = SSMSpec(
                n_latent=n,
                n_manifest=n,
                drift_spec=linear_drift_spec(
                    n_latent=n,
                    drift_diag_mask=full_diagonal_mask(n),
                    drift_offdiag_mask=full_drift_offdiag_mask(n),
                    drift_template=jnp.zeros((n, n)),
                    cint_mask=zero_vector_mask(n),
                    cint_template=jnp.zeros(n),
                ),
                diffusion_block=DiffusionBlockSpec(
                    n_latent=n,
                    diffusion_chol_mask=full_cholesky_mask(n),
                    diffusion_chol_template=jnp.eye(n),
                ),
                lambda_block=SparseMatrixBlockSpec(
                    n_rows=n,
                    n_cols=n,
                    mask=zero_loading_mask(n, n),
                    template=jnp.eye(n),
                    free_site_name="lambda_free",
                    det_site_name="lambda",
                ),
                manifest_means_block=SparseVectorBlockSpec(
                    n=n,
                    mask=zero_vector_mask(n),
                    template=jnp.zeros(n),
                    free_site_name="manifest_means_free",
                    det_site_name="manifest_means",
                ),
                manifest_chol_block=ManifestCholBlockSpec(
                    n_manifest=n,
                    diag_mask=full_diagonal_mask(n),
                    template=jnp.zeros((n, n)),
                ),
                t0_means_block=SparseVectorBlockSpec(
                    n=n,
                    mask=full_vector_mask(n),
                    template=jnp.zeros(n),
                    free_site_name="t0_means_free",
                    det_site_name="t0_means",
                ),
                t0_chol_block=T0CholBlockSpec(
                    n_latent=n,
                    diag_mask=full_diagonal_mask(n),
                    correlation_mask=strict_lower_triangle_mask(n),
                    template=jnp.eye(n),
                ),
                input_effect_block=SparseMatrixBlockSpec(
                    n_rows=n,
                    n_cols=0,
                    mask=np.zeros((n, 0), dtype=bool),
                    template=jnp.zeros((n, 0)),
                    free_site_name="input_effect_free",
                    det_site_name="input_effect",
                ),
                static_state_sd_block=SparseVectorBlockSpec(
                    n=0,
                    mask=np.zeros(0, dtype=bool),
                    template=jnp.zeros(0),
                    free_site_name="static_state_sd_free",
                    det_site_name="static_state_sds",
                ),
            )
            spec, priors, _bindings, _diagnostics, _edge_lag_days = compile_ssm_inputs_from_spec(
                ssm_spec=spec,
                priors=dump_prior_payloads(self._priors),
                causal_spec=self._causal_spec,
            )
        else:
            spec, priors = self.compile_inputs()

        spec = hydrate_discrete_manifest_metadata(spec, X)
        validate_observation_support(spec, X)

        prior_runtime_bundle = self._prior_runtime_bundle
        if prior_runtime_bundle is None and self._compiled_prior_semantics is not None:
            prior_runtime_bundle = self._load_prior_runtime_bundle(self._compiled_prior_semantics)

        self._model = SSMModel(
            spec,
            priors,
            prior_runtime_bundle=prior_runtime_bundle,
        )
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
        if not self.has_model:
            self.build_model(X, y)

        observations, times, _manifest_names = self.prepare_fit_inputs(X)
        self.model.set_transition_inputs(self.prepare_transition_inputs(X))
        return self.fit_prepared(observations, times, **kwargs)

    def fit_prepared(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        **kwargs: Any,
    ) -> InferenceResult:
        """Fit the built model from precomputed observation/time arrays."""
        if not self.has_model:
            raise ValueError("Model must be built before fitting prepared inputs")

        sampler_config = {**self._sampler_config, **kwargs}
        method = sampler_config.get("method", "aux_kalman_mcmc")
        fit_kwargs = {k: v for k, v in sampler_config.items() if k != "method"}

        result = fit(
            self.model,
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
        try:
            spec = self.spec
        except ValueError:
            spec = None

        if spec is not None and spec.manifest_names:
            manifest_cols = spec.manifest_names
        else:
            manifest_cols = default_manifest_columns(X)

        manifest_centered = (
            list(spec.manifest_centered)
            if spec is not None and spec.manifest_centered is not None
            else None
        )
        X = _center_manifest_columns(X, manifest_cols, manifest_centered)

        observations = jnp.array(X.select(manifest_cols).to_numpy(), dtype=jnp.float64)

        if "time" in X.columns:
            times = jnp.array(X["time"].to_numpy(), dtype=jnp.float64)
        else:
            times = jnp.arange(X.height, dtype=jnp.float64)

        return observations, times, manifest_cols

    def prepare_transition_inputs(self, X: pl.DataFrame) -> jnp.ndarray | None:
        """Extract known inputs in compiled input order and align them to transitions."""
        spec = self.spec
        input_names = list(spec.input_names or [])
        if not input_names:
            return None

        source_indicators = list(spec.input_source_indicators or [])
        scales = [float(scale) for scale in (spec.input_scales or [])]
        policies = list(spec.input_missing_policies or [])
        missing_sources = [name for name in source_indicators if name not in X.columns]
        if missing_sources:
            raise ValueError(
                f"Known input source indicators are absent from the model data: {missing_sources}"
            )

        columns: list[jnp.ndarray] = []
        for source_indicator, scale, policy in zip(
            source_indicators,
            scales,
            policies,
            strict=True,
        ):
            expr = pl.col(source_indicator).cast(pl.Float64, strict=False)
            if policy == "zero":
                filled = X.select(expr.fill_null(0.0).alias(source_indicator))
            elif policy == "forward_fill":
                filled = X.select(
                    expr.fill_null(strategy="forward").fill_null(0.0).alias(source_indicator)
                )
            else:
                raise ValueError(f"Unsupported known-input missing policy: {policy!r}")
            columns.append(
                jnp.asarray(filled[source_indicator].to_numpy(), dtype=jnp.float64) / scale
            )

        raw_inputs = jnp.stack(columns, axis=1)
        if raw_inputs.shape[0] <= 1:
            return raw_inputs
        return jnp.concatenate([raw_inputs[:1], raw_inputs[:-1]], axis=0)

    def sample_prior_predictive(self, samples: int = 500, times: jnp.ndarray | None = None) -> Any:
        """Sample from the prior predictive distribution.

        Args:
            samples: Number of samples
            times: Optional time points; defaults to arange(10)

        Returns:
            Prior predictive samples
        """
        prepared_times = getattr(self, "_prepared_times", None)
        prepared_inputs = getattr(self, "_prepared_transition_inputs", None)
        prepared_support = getattr(self, "_prepared_observation_support", None)
        prepared_mask = getattr(self, "_prepared_observation_mask", None)

        if times is None:
            if prepared_times is not None:
                times = prepared_times
            else:
                times = jnp.arange(10, dtype=jnp.float64)

        use_prepared_schedule = (
            prepared_times is not None
            and prepared_mask is not None
            and prepared_times.shape == times.shape
            and bool(jnp.allclose(prepared_times, times))
        )
        observation_support = prepared_support if use_prepared_schedule else None
        observation_mask = prepared_mask if use_prepared_schedule else None
        transition_inputs = prepared_inputs if use_prepared_schedule else None

        try:
            spec = self.spec
        except ValueError:
            if self._model_spec is not None:
                spec, _priors = self.compile_inputs()
            else:
                raise ValueError(
                    "Cannot sample prior predictive without an SSM specification"
                ) from None

        from nof1_causal_lab.models.ssm.inference.targets.observation_families import (
            any_family_needs_level_metadata,
        )
        from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
            sample_prior_predictive_from_priors,
            sample_prior_predictive_from_runtime,
        )

        needs_hydration = any_family_needs_level_metadata(spec.manifest_dists)
        if needs_hydration and spec.manifest_level_counts is None:
            raise ValueError(
                "Prior predictive for ordered/categorical emissions requires hydrated "
                "manifest_level_counts. Build the model with data first."
            )

        if self._compiled_prior_semantics is not None:
            runtime = self._load_prior_runtime_bundle(self._compiled_prior_semantics)
            return sample_prior_predictive_from_runtime(
                spec,
                runtime,
                times,
                observation_support=observation_support,
                observation_mask=observation_mask,
                transition_inputs=transition_inputs,
                num_samples=samples,
            )

        compiled_spec, priors = self.compile_inputs()
        try:
            runtime_spec = self.spec
        except ValueError:
            runtime_spec = compiled_spec
        return sample_prior_predictive_from_priors(
            runtime_spec,
            priors,
            times,
            observation_support=observation_support,
            observation_mask=observation_mask,
            transition_inputs=transition_inputs,
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
        from nof1_causal_lab.models.ssm.compile.artifact import build_compiled_ssm_builder

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


def prepare_wide_model_runtime(
    wide_data: pl.DataFrame,
    compiled_ssm: dict | None = None,
    sampler_config: dict | None = None,
    builder: SSMModelBuilder | None = None,
    observation_data: pl.DataFrame | None = None,
) -> PreparedModelRuntime:
    """Build or reuse a builder from wide data and extract fit-ready arrays."""
    if builder is None:
        if compiled_ssm is None:
            raise ValueError("Either builder or compiled_ssm must be provided")
        builder = build_ssm_builder(
            wide_data=wide_data,
            sampler_config=sampler_config,
            compiled_ssm=compiled_ssm,
        )
    elif not builder.has_model:
        builder.build_model(wide_data)

    builder_spec = builder.spec
    manifest_names = (
        list(builder_spec.manifest_names)
        if builder_spec.manifest_names
        else default_manifest_columns(wide_data)
    )
    wide_data = augment_wide_data_with_support_boundaries(
        observation_data,
        wide_data,
        manifest_names,
    )
    observations, times, manifest_names = builder.prepare_fit_inputs(wide_data)
    transition_inputs = builder.prepare_transition_inputs(wide_data)
    observation_support = compile_observation_support_runtime(
        observation_data,
        wide_data,
        manifest_names,
    )
    model_obj = builder.model
    model_obj.set_observation_support(observation_support)
    model_obj.set_transition_inputs(transition_inputs)
    spec_obj = builder.spec
    inference_structure = plan_inference_structure(
        spec_obj,
        observation_support=observation_support,
        method_override=(sampler_config or {}).get("method"),
        n_timepoints=int(times.shape[0]),
    )
    builder._prepared_times = times
    builder._prepared_transition_inputs = transition_inputs
    builder._prepared_observation_mask = ~jnp.isnan(observations)
    builder._prepared_observation_support = observation_support
    if observation_support is not None and observation_support.requires_interval_summary_handling:
        interval_summary_desc = ", ".join(
            f"{name} ({operator})"
            for name, operator, support_kind in zip(
                observation_support.manifest_names,
                observation_support.summary_operators,
                observation_support.support_kinds,
                strict=False,
            )
            if support_kind == "interval" and operator is not None
        )
        logger.info(
            "Prepared runtime compiled support-aware observation semantics for %s.",
            interval_summary_desc,
        )
    return PreparedModelRuntime(
        builder=builder,
        model=model_obj,
        spec=spec_obj,
        parameter_layout=model_obj.parameter_layout,
        wide_data=wide_data,
        observation_data=observation_data,
        observation_support=observation_support,
        inference_structure=inference_structure,
        observations=observations,
        times=times,
        transition_inputs=transition_inputs,
        manifest_names=manifest_names,
    )


def prepare_model_runtime(
    data_for_model: pl.DataFrame,
    compiled_ssm: dict | None = None,
    sampler_config: dict | None = None,
    builder: SSMModelBuilder | None = None,
) -> PreparedModelRuntime:
    """Canonical entry point for preparing stage data for model work."""
    return prepare_wide_model_runtime(
        pivot_to_wide(data_for_model),
        compiled_ssm=compiled_ssm,
        sampler_config=sampler_config,
        builder=builder,
        observation_data=data_for_model,
    )
