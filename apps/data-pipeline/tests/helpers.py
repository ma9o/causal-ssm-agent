"""Shared test helpers (non-fixtures).

These are utilities that can be imported directly into test modules.
For fixtures, see conftest.py.
"""

import asyncio

import jax.numpy as jnp

from causal_ssm_agent.orchestrator.stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4Plan,
)


def _run(coro):
    """Run an async function synchronously for testing."""
    return asyncio.run(coro)


def make_mock_generate(responses: list[str]):
    """Create a mock generate function that returns predefined responses.

    When tools are provided, the mock simulates the real generate loop by
    calling the first tool with the canned response (as if the LLM produced
    a tool call). This populates tool capture dicts so callers that rely on
    captured results (e.g. proxy validation) work correctly in tests.

    Args:
        responses: List of JSON strings to return in order

    Returns:
        Async function matching GenerateFn signature
    """
    call_count = [0]  # Use list to allow mutation in closure

    async def mock_generate(
        messages: list, tools: list | None, follow_ups: list[str] | None
    ) -> str:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        response = responses[idx]

        # When tools are provided, call the first tool with the response
        # to populate its capture dict (mirrors the real generate loop).
        if tools:
            tool = tools[0]
            # Find the first required parameter name from the tool's schema
            props = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])
            param_name = required[0] if required else next(iter(props), None)
            if param_name:
                await tool(**{param_name: response})
            else:
                await tool(response)

        return response

    return mock_generate


def make_stage4_plan(
    *,
    model_blocks: tuple[Stage4FrontierBlock, ...] = (),
    review_block: Stage4FrontierBlock | None = None,
    prior_blocks: tuple[Stage4FrontierBlock, ...] = (),
) -> Stage4Plan:
    """Build a minimal Stage 4 plan for focused unit tests."""
    all_blocks = (
        *model_blocks,
        *((review_block,) if review_block is not None else ()),
        *prior_blocks,
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
        blocks_by_id=blocks_by_id,
        parameter_to_block_id=parameter_to_block_id,
        indicator_to_decision_block_id=indicator_to_decision_block_id,
        indicator_to_measurement_block_id=indicator_to_measurement_block_id,
    )


def assert_recovery_ci(
    samples: jnp.ndarray,
    true_value: float,
    param_name: str,
    transform=None,
    q_low: float = 5.0,
    q_high: float = 95.0,
):
    """Assert that true_value falls within the [q_low, q_high] percentile CI.

    Args:
        samples: 1D array of posterior samples.
        true_value: Ground truth value.
        param_name: Name for error message.
        transform: Optional transform to apply to samples (e.g. lambda s: -jnp.abs(s)).
        q_low: Lower percentile (default 5 for 90% CI).
        q_high: Upper percentile (default 95 for 90% CI).
    """
    if transform is not None:
        samples = transform(samples)
    lo = float(jnp.percentile(samples, q_low))
    hi = float(jnp.percentile(samples, q_high))
    assert lo <= true_value <= hi, (
        f"{param_name} {true_value:.2f} outside {q_high - q_low:.0f}% CI [{lo:.3f}, {hi:.3f}]"
    )
