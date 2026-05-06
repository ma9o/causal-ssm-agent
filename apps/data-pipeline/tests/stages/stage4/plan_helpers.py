"""Stage 4 plan builders for tests."""

from causal_ssm_agent.flows.stages.stage4.agentic.stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4Plan,
    Stage4RepairTopology,
)


def make_stage4_plan(
    *,
    model_blocks: tuple[Stage4FrontierBlock, ...] = (),
    review_block: Stage4FrontierBlock | None = None,
    prior_blocks: tuple[Stage4FrontierBlock, ...] = (),
    prior_review_block: Stage4FrontierBlock | None = None,
) -> Stage4Plan:
    """Build a minimal Stage 4 plan for focused unit tests."""
    all_blocks = (
        *model_blocks,
        *((review_block,) if review_block is not None else ()),
        *prior_blocks,
        *((prior_review_block,) if prior_review_block is not None else ()),
    )
    blocks_by_id = {block.id: block for block in all_blocks}
    parameter_to_block_id: dict[str, str] = {}
    indicator_to_decision_block_id: dict[str, str] = {}
    indicator_to_measurement_block_id: dict[str, str] = {}

    for block in prior_blocks:
        for parameter_name in block.parameter_names:
            parameter_to_block_id.setdefault(parameter_name, block.id)
        if block.kind == "measurement_prior":
            for indicator_name in block.variable_names:
                indicator_to_measurement_block_id[indicator_name] = block.id

    for block in model_blocks:
        for parameter_name in block.parameter_names:
            parameter_to_block_id.setdefault(parameter_name, block.id)
        if block.kind == "indicator_decision":
            for indicator_name in block.variable_names:
                indicator_to_decision_block_id[indicator_name] = block.id

    return Stage4Plan(
        model_blocks=model_blocks,
        review_block=review_block,
        prior_blocks=prior_blocks,
        prior_review_block=prior_review_block,
        blocks_by_id=blocks_by_id,
        repair_topology=Stage4RepairTopology(
            parameter_to_block_id=parameter_to_block_id,
            indicator_to_decision_block_id=indicator_to_decision_block_id,
            indicator_to_measurement_block_id=indicator_to_measurement_block_id,
        ),
    )
