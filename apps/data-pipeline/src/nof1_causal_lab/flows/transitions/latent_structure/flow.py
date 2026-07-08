"""latent-structure: Latent Structure Proposal."""

from nof1_causal_lab.flows.llm_transition_runtime import make_llm_transition_runner
from nof1_causal_lab.utils.config import get_config

from .run import run_latent_structure

propose_latent_structure = make_llm_transition_runner(
    context_id="latent-structure",
    orchestrator_fn=run_latent_structure,
    profile_llm_getter=lambda: get_config().structure_proposal.llm,
    max_tool_turns_getter=lambda: get_config().structure_proposal.latent_max_tool_turns,
    payload_builder=lambda result: {"latent_structure": result.latent_structure},
)
