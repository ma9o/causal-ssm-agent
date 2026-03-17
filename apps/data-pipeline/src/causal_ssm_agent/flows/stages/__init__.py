"""Pipeline stages."""

from .persist import (
    persist_web_result,
)
from .stage0_ingest import (
    agentic_ingest,
)
from .stage1a_latent import (
    propose_latent_model,
)
from .stage1b_measurement import (
    build_causal_spec,
    propose_measurement_with_identifiability_fix,
)
from .stage2_extract import (
    stage2_extraction_flow,
)
from .stage3_validation import (
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
    PreparedModelRuntime,
    fit_model,
    prepare_model_runtime,
    run_interventions,
    run_power_scaling,
    run_ppc,
)

__all__ = [
    # Stage 0
    "agentic_ingest",
    # Stage 1a
    "propose_latent_model",
    # Stage 1b
    "propose_measurement_with_identifiability_fix",
    "build_causal_spec",
    # Stage 2
    "stage2_extraction_flow",
    # Stage 3: Validate
    "validate_extraction",
    # Persistence
    "persist_web_result",
    # Stage 4
    "stage4_orchestrated_flow",
    # Stage 4b
    "parametric_id_task",
    "stage4b_parametric_id_flow",
    # Stage 5
    "PreparedModelRuntime",
    "fit_model",
    "prepare_model_runtime",
    "run_interventions",
    "run_ppc",
    "run_power_scaling",
]
