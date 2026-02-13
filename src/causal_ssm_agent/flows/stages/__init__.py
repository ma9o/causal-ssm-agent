"""Pipeline stages."""

from .stage0_preprocess import (
    preprocess_raw_input,
)
from .stage1a_latent import (
    propose_latent_model,
)
from .stage1b_measurement import (
    build_causal_spec,
    load_orchestrator_chunks,
    propose_measurement_with_identifiability_fix,
)
from .stage2_workers import (
    load_worker_chunks,
    populate_indicators,
)
from .stage3_validation import (
    aggregate_measurements,
    combine_worker_results,
    validate_extraction,
)
from .stage4_model import (
    stage4_orchestrated_flow,
)
from .stage4b_parametric_id import (
    parametric_id_task,
    stage4b_parametric_id_flow,
)
from .stage5_inference import (
    fit_model,
    run_interventions,
    run_power_scaling,
    run_ppc,
)

__all__ = [
    # Stage 0
    "preprocess_raw_input",
    # Stage 1a
    "propose_latent_model",
    # Stage 1b
    "load_orchestrator_chunks",
    "propose_measurement_with_identifiability_fix",
    "build_causal_spec",
    # Stage 2: Extract
    "load_worker_chunks",
    "populate_indicators",
    # Stage 3: Validate & Aggregate
    "aggregate_measurements",
    "combine_worker_results",
    "validate_extraction",
    # Stage 4
    "stage4_orchestrated_flow",
    # Stage 4b
    "parametric_id_task",
    "stage4b_parametric_id_flow",
    # Stage 5
    "fit_model",
    "run_interventions",
    "run_ppc",
    "run_power_scaling",
]
