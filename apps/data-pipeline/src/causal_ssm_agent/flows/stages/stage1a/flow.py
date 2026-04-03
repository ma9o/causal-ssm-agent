"""Stage 1a: Latent Model Proposal (Prefect wrapper)."""

from causal_ssm_agent.flows.llm_stage_task import make_llm_stage_task
from causal_ssm_agent.orchestrator.stage1a import run_stage1a
from causal_ssm_agent.utils.config import get_config

propose_latent_model = make_llm_stage_task(
    stage_id="stage-1a",
    orchestrator_fn=run_stage1a,
    model_name_getter=lambda: get_config().stage1_structure_proposal.model,
    max_tool_turns_getter=lambda: get_config().stage1_structure_proposal.stage1a_max_tool_turns,
    payload_builder=lambda result: {"latent_model": result.latent_model},
    task_options={
        "retries": 2,
        "retry_delay_seconds": 30,
    },
)
