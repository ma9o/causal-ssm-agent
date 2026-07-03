"""Stage 1b: Measurement Model Proposal."""

from nof1_causal_lab.flows.llm_stage_runtime import make_llm_stage_runner
from nof1_causal_lab.utils.config import get_config

from .assemble import build_causal_spec as _build_causal_spec_core
from .run import run_stage1b


def build_causal_spec(
    latent_model: dict,
    measurement_model: dict,
    identifiability_status: dict | None = None,
    known_inputs: list[dict] | None = None,
) -> dict:
    """Combine latent and measurement models into full CausalSpec with identifiability."""
    return _build_causal_spec_core(
        latent_model,
        measurement_model,
        identifiability_status,
        known_inputs=known_inputs,
    )


propose_measurement_with_identifiability_fix = make_llm_stage_runner(
    stage_id="stage-1b",
    orchestrator_fn=run_stage1b,
    stage_llm_getter=lambda: get_config().stage1_structure_proposal.llm,
    max_tool_turns_getter=lambda: get_config().stage1_structure_proposal.stage1b_max_tool_turns,
    payload_builder=lambda result: {"causal_spec": result.causal_spec},
)
