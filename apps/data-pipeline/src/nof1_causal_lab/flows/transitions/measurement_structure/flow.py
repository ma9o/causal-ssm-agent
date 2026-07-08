"""measurement-structure: Measurement Structure Proposal."""

from nof1_causal_lab.flows.llm_transition_runtime import make_llm_transition_runner
from nof1_causal_lab.utils.config import get_config

from .assemble import build_causal_design as _build_causal_design_core
from .run import run_measurement_structure


def build_causal_design(
    latent_structure: dict,
    measurement_structure: dict,
    identifiability_status: dict | None = None,
    known_inputs: list[dict] | None = None,
) -> dict:
    """Combine latent and measurement structures into full CausalDesign with identifiability."""
    return _build_causal_design_core(
        latent_structure,
        measurement_structure,
        identifiability_status,
        known_inputs=known_inputs,
    )


propose_measurement_structure = make_llm_transition_runner(
    context_id="measurement-structure",
    orchestrator_fn=run_measurement_structure,
    profile_llm_getter=lambda: get_config().structure_proposal.llm,
    max_tool_turns_getter=lambda: get_config().structure_proposal.measurement_max_tool_turns,
    payload_builder=lambda result: {"measurement_structure": result.measurement_structure},
)
