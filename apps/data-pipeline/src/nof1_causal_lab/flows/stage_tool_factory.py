"""Shared tool factory for interactive stage grounding."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from causal_ssm_agent.flows import get_prefect_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_prefect_logger(__name__)


def make_stage_tool(
    name: str,
    description: str,
    param_name: str,
    param_description: str,
    compute_fn: Callable[[dict], tuple[dict | None, str]],
    success_feedback: str = "VALID",
    capture_when: Callable[[dict | None, str], bool] | None = None,
) -> tuple[Any, dict]:
    """Create a fat tool for pipeline use wrapping a compute function.

    The returned Tool calls compute_fn on the parsed JSON input.
    On success (stage_output is not None), the capture dict is updated.
    The tool returns the feedback string to the LLM.

    Args:
        name: Tool name
        description: Tool description
        param_name: Name of the JSON string parameter
        param_description: Description of the parameter
        compute_fn: (data_dict) -> (stage_output | None, feedback_str)
        success_feedback: The feedback string that triggers stop_on_success

    Returns:
        (Tool, capture_dict)
    """
    from causal_ssm_agent.utils.openrouter_client import Tool

    capture: dict = {}

    async def _execute(**kwargs: str) -> str:
        try:
            data = json.loads(kwargs[param_name])
        except json.JSONDecodeError as e:
            logger.warning("[%s] JSON parse error: %s", name, e)
            return f"JSON parse error: {e}"

        t0 = time.monotonic()
        stage_output, feedback = compute_fn(data)
        elapsed = time.monotonic() - t0
        is_success = feedback == success_feedback

        should_capture = (
            capture_when(stage_output, feedback) if capture_when else stage_output is not None
        )
        if should_capture and stage_output is not None:
            capture.update(stage_output)
        if is_success:
            logger.info("[%s] grounding passed (%.1fs)", name, elapsed)
        else:
            preview = feedback[:200].replace("\n", " ")
            logger.info("[%s] grounding rejected (%.1fs): %s", name, elapsed, preview)

        return feedback

    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                param_name: {"type": "string", "description": param_description},
            },
            "required": [param_name],
            "additionalProperties": False,
        },
        execute=_execute,
        stop_on_success=True,
        success_output=success_feedback,
    ), capture
