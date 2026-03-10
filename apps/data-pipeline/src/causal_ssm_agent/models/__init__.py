"""NumPyro state-space model builders and validation."""

from .prior_predictive import (
    format_validation_report,
    validate_prior_predictive,
)
from .ssm import SSMModel, SSMPriors, SSMSpec
from .ssm_builder import SSMModelBuilder
from .ssm_compiler import (
    build_compiled_ssm_builder,
    compile_ssm_artifact,
    deserialize_ssm_priors,
    deserialize_ssm_spec,
    trial_compile_measurement_model,
    trial_compile_model_spec,
    validate_measurement_model_for_compilation,
    validate_model_spec_for_compilation,
)

__all__ = [
    # State-space model
    "SSMModel",
    "SSMPriors",
    "SSMSpec",
    "SSMModelBuilder",
    "compile_ssm_artifact",
    "build_compiled_ssm_builder",
    "deserialize_ssm_spec",
    "deserialize_ssm_priors",
    "validate_measurement_model_for_compilation",
    "validate_model_spec_for_compilation",
    "trial_compile_measurement_model",
    "trial_compile_model_spec",
    # Validation
    "validate_prior_predictive",
    "format_validation_report",
]
