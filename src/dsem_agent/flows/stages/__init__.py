"""Pipeline stages."""

from .stage1a_latent import (
    propose_latent_model,
)
from .stage1b_measurement import (
    build_dsem_model,
    load_orchestrator_chunks,
    propose_measurement_with_identifiability_fix,
)
from .stage2_workers import (
    aggregate_measurements,
    load_worker_chunks,
    populate_indicators,
)
from .stage3_validation import (
    revalidate_dag_for_missing_proxies,
    validate_extraction,
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
    "propose_latent_model",
    # Stage 1b
    "load_orchestrator_chunks",
    "propose_measurement_with_identifiability_fix",
    "build_dsem_model",
    # Stage 2
    "load_worker_chunks",
    "populate_indicators",
    "aggregate_measurements",
    # Stage 3
    "validate_extraction",
    "revalidate_dag_for_missing_proxies",
    # Stage 4
    "specify_model",
    "elicit_priors",
    # Stage 5
    "fit_model",
    "run_interventions",
]
