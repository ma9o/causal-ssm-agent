"""Pipeline stages."""

from .stage1a_structural import (
    propose_structural_model,
)
from .stage1b_measurement import (
    build_dsem_model,
    load_orchestrator_chunks,
    propose_measurement_model,
)
from .stage2_workers import (
    aggregate_measurements,
    load_worker_chunks,
    populate_indicators,
)
from .stage3_identifiability import (
    check_identifiability,
    run_sensitivity_analysis,
)
from .stage4_model import (
    elicit_priors,
    specify_model,
)
from .stage5_inference import (
    fit_model,
    run_interventions,
)

__all__ = [
    # Stage 1a
    "propose_structural_model",
    # Stage 1b
    "load_orchestrator_chunks",
    "propose_measurement_model",
    "build_dsem_model",
    # Stage 2
    "load_worker_chunks",
    "populate_indicators",
    "aggregate_measurements",
    # Stage 3
    "check_identifiability",
    "run_sensitivity_analysis",
    # Stage 4
    "specify_model",
    "elicit_priors",
    # Stage 5
    "fit_model",
    "run_interventions",
]
