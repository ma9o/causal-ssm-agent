"""Stage 1a: Latent Model Proposal."""

from nof1_causal_lab.flows.llm_stage_runtime import make_llm_stage_runner
from nof1_causal_lab.utils.config import get_config

from .run import run_stage1a

propose_latent_model = make_llm_stage_runner(
    stage_id="stage-1a",
    orchestrator_fn=run_stage1a,
    stage_llm_getter=lambda: get_config().stage1_structure_proposal.llm,
    max_tool_turns_getter=lambda: get_config().stage1_structure_proposal.stage1a_max_tool_turns,
    payload_builder=lambda result: {"latent_model": result.latent_model},
)
