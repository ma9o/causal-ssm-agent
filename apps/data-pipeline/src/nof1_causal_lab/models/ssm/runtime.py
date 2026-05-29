"""Pure preparation helpers for executable SSM models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import polars as pl

from nof1_causal_lab.models.ssm import (
    PriorRegistry,
    SSMModel,
    SSMParameterLayout,
    SSMSpec,
    fit,
)
from nof1_causal_lab.models.ssm.compile.common import dump_prior_payloads
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_ssm_inputs_from_model_spec,
    compile_ssm_inputs_from_spec,
)
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
from nof1_causal_lab.utils.data import pivot_to_wide

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.inference import InferenceResult
    from nof1_causal_lab.workers.schemas_prior import PriorProposal

logger = logging.getLogger(__name__)


def _center_manifest_columns(
    wide_data: pl.DataFrame,
    manifest_cols: list[str],
    manifest_centered: list[bool] | None,
) -> pl.DataFrame:
    """Apply deterministic centering to manifest columns marked centered."""
    if manifest_centered is None or not any(manifest_centered):
        return wide_data

    centered_exprs = []
    for manifest_name, centered in zip(manifest_cols, manifest_centered, strict=False):
        base_expr = pl.col(manifest_name).cast(pl.Float64, strict=False)
        if centered:
            centered_exprs.append((base_expr - base_expr.mean()).alias(manifest_name))
        else:
            centered_exprs.append(base_expr.alias(manifest_name))
    passthrough = [col_name for col_name in wide_data.columns if col_name not in set(manifest_cols)]
    return wide_data.select(*passthrough, *centered_exprs)


@dataclass
class PreparedModelRuntime:
    """Canonical prepared runtime context shared by validation and inference."""

    model: SSMModel
    spec: SSMSpec
    parameter_layout: SSMParameterLayout
    sampler_config: dict[str, Any]
    wide_data: pl.DataFrame
    observation_data: pl.DataFrame | None
    observation_support: ObservationSupportRuntime | None
    inference_structure: InferenceStructurePlan
    observations: jnp.ndarray
    times: jnp.ndarray
    transition_inputs: jnp.ndarray | None
    manifest_names: list[str]


def get_default_sampler_config() -> dict[str, Any]:
    """Return default sampler configuration from config.yaml."""
    from nof1_causal_lab.utils.config import get_config

    return get_config().inference.to_sampler_config()


def compile_model_inputs(
    *,
    model_spec: ModelSpec | dict | None = None,
    priors: dict[str, PriorProposal] | dict[str, dict] | None = None,
    ssm_spec: SSMSpec | None = None,
    prior_registry: PriorRegistry | None = None,
    causal_spec: dict | None = None,
) -> tuple[SSMSpec, PriorRegistry, list[dict[str, object]]]:
    """Compile user-facing or direct SSM inputs into executable model inputs."""
    if model_spec is not None and (ssm_spec is not None or prior_registry is not None):
        raise ValueError(
            "Compile either ModelSpec-driven inputs or direct SSMSpec inputs, not both."
        )
    if model_spec is None and ssm_spec is not None and causal_spec is not None:
        raise ValueError(
            "Do not pass causal_spec alongside a direct SSMSpec. Compile from ModelSpec + "
            "CausalSpec or use a compiled artifact so the causal structure is encoded "
            "explicitly in the spec masks."
        )

    if model_spec is not None:
        spec, resolved_priors, bindings, _diagnostics, _edge_lag_days = (
            compile_ssm_inputs_from_model_spec(
                model_spec=model_spec,
                priors=dump_prior_payloads(priors or {}),
                causal_spec=causal_spec,
            )
        )
    elif ssm_spec is not None:
        spec, resolved_priors, bindings, _diagnostics, _edge_lag_days = (
            compile_ssm_inputs_from_spec(
                ssm_spec=ssm_spec,
                priors=dump_prior_payloads(priors or {}),
                prior_registry=prior_registry,
                model_spec=model_spec,
                causal_spec=causal_spec,
            )
        )
    else:
        raise ValueError("Model construction requires either model_spec or ssm_spec.")

    return spec, resolved_priors, bindings


def build_ssm_model(
    wide_data: pl.DataFrame,
    *,
    model_spec: ModelSpec | dict | None = None,
    priors: dict[str, PriorProposal] | dict[str, dict] | None = None,
    ssm_spec: SSMSpec | None = None,
    prior_registry: PriorRegistry | None = None,
    compiled_prior_semantics: dict | None = None,
    prior_runtime_bundle: PriorRuntimeBundle | None = None,
    causal_spec: dict | None = None,
    parameter_bindings: list[dict[str, Any]] | None = None,
) -> SSMModel:
    """Build a live ``SSMModel`` from compiled inputs and wide data."""
    if wide_data.is_empty():
        raise ValueError("Cannot build SSM model from empty data")

    spec, resolved_priors, bindings = compile_model_inputs(
        model_spec=model_spec,
        priors=priors,
        ssm_spec=ssm_spec,
        prior_registry=prior_registry,
        causal_spec=causal_spec,
    )
    spec = hydrate_discrete_manifest_metadata(spec, wide_data)
    validate_observation_support(spec, wide_data)

    runtime_bundle = prior_runtime_bundle
    if runtime_bundle is None and compiled_prior_semantics is not None:
        runtime_bundle = load_prior_runtime_bundle(compiled_prior_semantics)

    model = SSMModel(
        spec,
        resolved_priors,
        prior_runtime_bundle=runtime_bundle,
    )
    model.parameter_bindings = list(parameter_bindings or bindings)
    return model


def prepare_fit_inputs(
    spec: SSMSpec,
    wide_data: pl.DataFrame,
) -> tuple[jnp.ndarray, jnp.ndarray, list[str], pl.DataFrame]:
    """Extract observations, times, manifest order, and centered wide data."""
    manifest_cols = (
        list(spec.manifest_names) if spec.manifest_names else default_manifest_columns(wide_data)
    )
    manifest_centered = list(spec.manifest_centered) if spec.manifest_centered is not None else None
    centered_data = _center_manifest_columns(wide_data, manifest_cols, manifest_centered)
    observations = jnp.array(centered_data.select(manifest_cols).to_numpy(), dtype=jnp.float64)
    if "time" in centered_data.columns:
        times = jnp.array(centered_data["time"].to_numpy(), dtype=jnp.float64)
    else:
        times = jnp.arange(centered_data.height, dtype=jnp.float64)
    return observations, times, manifest_cols, centered_data


def prepare_transition_inputs(spec: SSMSpec, wide_data: pl.DataFrame) -> jnp.ndarray | None:
    """Extract known inputs in compiled input order and align them to transitions."""
    input_names = list(spec.input_names or [])
    if not input_names:
        return None

    source_indicators = list(spec.input_source_indicators or [])
    scales = [float(scale) for scale in (spec.input_scales or [])]
    policies = list(spec.input_missing_policies or [])
    missing_sources = [name for name in source_indicators if name not in wide_data.columns]
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
            filled = wide_data.select(expr.fill_null(0.0).alias(source_indicator))
        elif policy == "forward_fill":
            filled = wide_data.select(
                expr.fill_null(strategy="forward").fill_null(0.0).alias(source_indicator)
            )
        else:
            raise ValueError(f"Unsupported known-input missing policy: {policy!r}")
        columns.append(jnp.asarray(filled[source_indicator].to_numpy(), dtype=jnp.float64) / scale)

    raw_inputs = jnp.stack(columns, axis=1)
    if raw_inputs.shape[0] <= 1:
        return raw_inputs
    return jnp.concatenate([raw_inputs[:1], raw_inputs[:-1]], axis=0)


def prepare_wide_model_runtime(
    wide_data: pl.DataFrame,
    *,
    compiled_ssm: dict | None = None,
    sampler_config: dict[str, Any] | None = None,
    model: SSMModel | None = None,
    observation_data: pl.DataFrame | None = None,
) -> PreparedModelRuntime:
    """Build or reuse an ``SSMModel`` and extract fit-ready arrays."""
    resolved_sampler_config = sampler_config or get_default_sampler_config()
    if model is None:
        if compiled_ssm is None:
            raise ValueError("Either model or compiled_ssm must be provided")
        from nof1_causal_lab.models.ssm.compile.artifact import build_model_from_compiled_artifact

        model = build_model_from_compiled_artifact(compiled_ssm, wide_data)

    spec = model.spec
    manifest_names = (
        list(spec.manifest_names) if spec.manifest_names else default_manifest_columns(wide_data)
    )
    wide_data = augment_wide_data_with_support_boundaries(
        observation_data,
        wide_data,
        manifest_names,
    )
    observations, times, manifest_names, wide_data = prepare_fit_inputs(spec, wide_data)
    transition_inputs = prepare_transition_inputs(spec, wide_data)
    observation_support = compile_observation_support_runtime(
        observation_data,
        wide_data,
        manifest_names,
    )
    model.set_observation_support(observation_support)
    model.set_transition_inputs(transition_inputs)
    inference_structure = plan_inference_structure(
        spec,
        observation_support=observation_support,
        method_override=resolved_sampler_config.get("method"),
        n_timepoints=int(times.shape[0]),
    )
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
        model=model,
        spec=spec,
        parameter_layout=model.parameter_layout,
        sampler_config=resolved_sampler_config,
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
    *,
    compiled_ssm: dict | None = None,
    sampler_config: dict[str, Any] | None = None,
    model: SSMModel | None = None,
) -> PreparedModelRuntime:
    """Canonical entry point for preparing stage data for model work."""
    return prepare_wide_model_runtime(
        pivot_to_wide(data_for_model),
        compiled_ssm=compiled_ssm,
        sampler_config=sampler_config,
        model=model,
        observation_data=data_for_model,
    )


def fit_prepared_model(runtime: PreparedModelRuntime, **kwargs: Any) -> InferenceResult:
    """Run public SSM inference against a prepared runtime."""
    sampler_config = {**runtime.sampler_config, **kwargs}
    method = sampler_config.get("method", "marginal_particle_gibbs")
    fit_kwargs = {key: value for key, value in sampler_config.items() if key != "method"}
    return fit(
        runtime.model,
        observations=runtime.observations,
        times=runtime.times,
        method=method,
        **fit_kwargs,
    )


def sample_prior_predictive(
    model: SSMModel,
    *,
    samples: int = 500,
    times: jnp.ndarray | None = None,
    observation_support: ObservationSupportRuntime | None = None,
    observation_mask: jnp.ndarray | None = None,
    transition_inputs: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from a live model and optional prepared schedule."""
    from nof1_causal_lab.models.ssm.inference.targets.observation_families import (
        any_family_needs_level_metadata,
    )
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
        sample_prior_predictive_from_runtime,
    )

    spec = model.spec
    if any_family_needs_level_metadata(spec.manifest_dists) and spec.manifest_level_counts is None:
        raise ValueError(
            "Prior predictive for ordered/categorical emissions requires hydrated "
            "manifest_level_counts. Build the model with data first."
        )

    if times is None:
        times = jnp.arange(10, dtype=jnp.float64)
    return sample_prior_predictive_from_runtime(
        spec,
        model.get_prior_runtime_bundle(),
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        transition_inputs=transition_inputs,
        num_samples=samples,
    )
