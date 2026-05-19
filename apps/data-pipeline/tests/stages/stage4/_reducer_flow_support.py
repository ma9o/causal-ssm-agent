"""Scripted-LLM machinery and reducer-flow scenario builders.

Used exclusively by ``test_reducer_flow.py``. Helpers shared with other
Stage 4 test files live in ``_support.py``.
"""

from types import SimpleNamespace
from typing import Any

from nof1_causal_lab.flows.stages.stage4.agentic.stage4_feedback import (
    make_stage4_grounding_result,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_navigation import (
    get_active_plan_block,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_orchestrator import (
    Stage4FrontierBlock,
    Stage4Plan,
)
from nof1_causal_lab.flows.stages.stage4.agentic.stage4_state import (
    Stage4Runtime,
)
from nof1_causal_lab.flows.stages.stage4.assembly import AssemblyValidation
from nof1_causal_lab.utils.llm import LLMTrace
from nof1_causal_lab.utils.openrouter_client import Tool
from tests.stages.stage4._support import _ORDINAL_LEVELS, _with_positive_indicator_polarity


def _make_stage4_global_repair_spec() -> dict:
    """Stage 4 spec with a correlation block and separate dynamics blocks."""
    constructs = [
        {
            "name": "activity",
            "role": "exogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "sleep",
            "role": "endogenous",
            "temporal_status": "time_varying",
            "is_outcome": True,
        },
    ]
    indicators = [
        {
            "name": "activity_vas",
            "construct_name": "activity",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Activity visual analog scale",
            "aggregation": "mean",
        },
        {
            "name": "sleep_quality",
            "construct_name": "sleep",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Sleep quality rating",
            "aggregation": "mean",
        },
    ]
    return _with_positive_indicator_polarity(
        {
            "latent": {
                "constructs": constructs,
                "edges": [{"cause": "activity", "effect": "sleep"}],
            },
            "measurement": {"model_clock": "1d", "indicators": indicators},
            "estimation": {
                "state_order": ["activity", "sleep"],
                "edges": [{"cause": "activity", "effect": "sleep"}],
                "induced_dependencies": [
                    {
                        "between": ["activity", "sleep"],
                        "kind": "initial_state_correlation",
                        "source_confounders": ["U"],
                    }
                ],
            },
        }
    )


def _make_stage4_two_effect_spec() -> dict:
    """Stage 4 spec with two sequential effect blocks."""
    constructs = [
        {
            "name": "activity",
            "role": "exogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "sleep",
            "role": "endogenous",
            "temporal_status": "time_varying",
        },
        {
            "name": "mood",
            "role": "endogenous",
            "temporal_status": "time_varying",
            "is_outcome": True,
        },
    ]
    edges = [
        {"cause": "activity", "effect": "sleep"},
        {"cause": "activity", "effect": "mood"},
    ]
    indicators = [
        {
            "name": "steps",
            "construct_name": "activity",
            "measurement_dtype": "count",
            "how_to_measure": "Daily step count",
            "aggregation": "sum",
        },
        {
            "name": "activity_vas",
            "construct_name": "activity",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Activity visual analog scale",
            "aggregation": "mean",
        },
        {
            "name": "sleep_quality",
            "construct_name": "sleep",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Sleep quality rating",
            "aggregation": "mean",
        },
        {
            "name": "mood_rating",
            "construct_name": "mood",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Mood rating",
            "aggregation": "mean",
        },
    ]
    return _with_positive_indicator_polarity(
        {
            "latent": {"constructs": constructs, "edges": edges},
            "measurement": {"model_clock": "1d", "indicators": indicators},
            "estimation": {
                "state_order": ["activity", "sleep", "mood"],
                "edges": edges,
                "induced_dependencies": [],
            },
        }
    )


def _make_stage4_no_model_block_spec() -> dict:
    """Stage 4 spec whose reducer starts directly in the prior phase."""
    constructs = [
        {
            "name": "sleep",
            "role": "endogenous",
            "temporal_status": "time_varying",
            "is_outcome": True,
        },
    ]
    indicators = [
        {
            "name": "sleep_quality",
            "construct_name": "sleep",
            "measurement_dtype": "ordinal",
            "ordinal_levels": list(_ORDINAL_LEVELS),
            "how_to_measure": "Sleep quality rating",
            "aggregation": "mean",
        },
    ]
    return _with_positive_indicator_polarity(
        {
            "latent": {"constructs": constructs, "edges": []},
            "measurement": {"model_clock": "1d", "indicators": indicators},
            "estimation": {
                "state_order": ["sleep"],
                "edges": [],
                "induced_dependencies": [],
            },
        }
    )


def _make_stub_grounding_result(stage_output: dict | None, feedback: str):
    """Wrap test grounding payloads in the typed Stage 4 grounding result."""
    validation = stage_output.get("validation") if isinstance(stage_output, dict) else None
    if validation is not None and getattr(validation, "compile_ok", True) is False:
        status = "compile_error"
    elif (
        validation is not None
        and getattr(validation, "pp_checked", False)
        and getattr(validation, "pp_valid", True) is False
    ):
        status = "prior_predictive_failure"
    elif "missing priors" in feedback.lower():
        status = "accepted_pending_priors"
    else:
        status = "accepted"
    return make_stage4_grounding_result(
        stage_output=stage_output,
        status=status,
        feedback=feedback,
        validation=validation,
        retain_for_next_prompt=feedback != "VALID",
        capture_stage_output=stage_output is not None
        and status in {"accepted", "accepted_pending_priors"},
    )


def _require_active_plan_block(plan: Stage4Plan, runtime: Stage4Runtime) -> Stage4FrontierBlock:
    """Fetch the active block and assert the runtime is currently promptable."""
    block = get_active_plan_block(plan, runtime)
    assert block is not None
    return block


def _current_stage4_state(current: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize optional grounding state payloads for test stubs."""
    return {} if current is None else current


def _activity_measurement_prior_bundle(
    *,
    lambda_sigma: float,
    lambda_reasoning: str,
    obs_sd_activity_vas_sigma: float = 0.5,
    obs_sd_steps_sigma: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Return a complete measurement block bundle for the mechanics fixtures."""
    return {
        "obs_sd_activity_vas": {
            "parameter": "obs_sd_activity_vas",
            "distribution": "HalfNormal",
            "params": {"sigma": obs_sd_activity_vas_sigma},
            "sources": [],
            "reasoning": "activity VAS measurement noise",
        },
        "obs_sd_steps": {
            "parameter": "obs_sd_steps",
            "distribution": "HalfNormal",
            "params": {"sigma": obs_sd_steps_sigma},
            "sources": [],
            "reasoning": "steps measurement noise",
        },
        "lambda_activity_vas_activity": {
            "parameter": "lambda_activity_vas_activity",
            "distribution": "HalfNormal",
            "params": {"sigma": lambda_sigma},
            "sources": [],
            "reasoning": lambda_reasoning,
        },
    }


def _activity_loading_prior_bundle(
    *,
    lambda_sigma: float,
    lambda_reasoning: str,
) -> dict[str, dict[str, Any]]:
    """Return only the activity loading prior for local repair prompts."""
    return {
        "lambda_activity_vas_activity": {
            "parameter": "lambda_activity_vas_activity",
            "distribution": "HalfNormal",
            "params": {"sigma": lambda_sigma},
            "sources": [],
            "reasoning": lambda_reasoning,
        },
    }


async def _await_string(awaitable) -> str:
    """Bridge `Awaitable[str]` helpers into `asyncio.run()` in tests."""
    return await awaitable


def _require_trace(trace_capture: dict[str, object]) -> LLMTrace:
    """Extract the accumulated trace from a generate capture."""
    trace = trace_capture["trace"]
    assert isinstance(trace, LLMTrace)
    return trace


class _ScriptedStage4AgentSession:
    """Test adapter that drives the current StageSessionFactory API."""

    def __init__(self, generate, *, system_prompt: str | None, tools: list[Tool], log_label: str):
        self._generate = generate
        self._messages: list[dict[str, str]] = []
        if system_prompt is not None:
            self._messages.append({"role": "system", "content": system_prompt})
        self._tools = tools
        self._log_label = log_label
        self._completion = ""
        self._trace = LLMTrace()

    async def turn(self, user_message: str):
        self._messages.append({"role": "user", "content": user_message})
        self._completion = await self._generate(
            self._messages,
            self._tools,
            label=self._log_label,
        )
        return SimpleNamespace(
            completion=self._completion,
            terminal_tool_name=None,
            terminal_tool_output=None,
            tool_calls_fired=[],
        )

    @property
    def result(self):
        return SimpleNamespace(
            completion=self._completion,
            trace=self._trace,
            terminal_tool_name=None,
            terminal_tool_output=None,
        )


class _ScriptedStage4OpenContext:
    def __init__(self, session: _ScriptedStage4AgentSession):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ScriptedStage4SessionFactory:
    def __init__(self, generate):
        self._generate = generate
        self.accumulated_trace = LLMTrace()

    def open(
        self,
        *,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        log_label: str | None = None,
    ):
        return _ScriptedStage4OpenContext(
            _ScriptedStage4AgentSession(
                self._generate,
                system_prompt=system_prompt,
                tools=tools or [],
                log_label=log_label or "stage-4",
            )
        )


def _make_stage4_session_factory(generate) -> _ScriptedStage4SessionFactory:
    return _ScriptedStage4SessionFactory(generate)


def _stub_stage4_repair_barrier_success(_plan, runtime, _deps) -> tuple[dict[str, str], ...]:
    runtime.domain.accepted.validation = AssemblyValidation(
        normalized_model_spec=runtime.domain.accepted.model_spec,
        compile_ok=True,
        pp_checked=True,
        pp_valid=True,
        diagnostics=[],
    )
    runtime.domain.repair_campaign = None
    runtime.domain.active_block_id = None
    runtime.domain.done = True
    return ({"block_id": "repair:barrier", "status": "accepted"},)


def _stage4_submit_tool_name(block_kind: str) -> str:
    """Return the primary submit-tool name for one Stage 4 block kind."""
    if block_kind == "model_configuration":
        return "submit_model_configuration"
    if block_kind == "indicator_decision":
        return "submit_indicator_choice"
    if block_kind == "global_review":
        return "submit_model_review"
    return "submit_prior_block"


def _stage4_submit_tool_args(submission: dict[str, object]) -> dict[str, object]:
    """Extract the direct tool arguments from a legacy block-keyed submission fixture."""
    proposal = submission.get("proposal")
    assert isinstance(proposal, dict)
    normalized: dict[str, object] = {}
    for key, value in proposal.items():
        assert isinstance(key, str)
        normalized[key] = value
    return normalized


def _make_scripted_stage4_generate(
    submissions: list[dict[str, object]],
    *,
    visited_blocks: list[str],
    visible_tools: list[list[str]],
):
    """Drive ``run_stage4()`` with scripted block-local submit-tool calls only."""
    turn_index = 0

    async def _generate(messages, tools, rewrite_messages=None, rewrite_tools=None, label=None):
        nonlocal turn_index
        del messages, rewrite_messages, rewrite_tools
        block_id = None
        if isinstance(label, str) and label.startswith("stage-4:"):
            block_id = label.removeprefix("stage-4:")
        if block_id == "model:configuration":
            submit_tool = next(tool for tool in tools if tool.name == "submit_model_configuration")
            feedback = await submit_tool(
                initialization_policy="stationary",
                observation_intercept_policy="free",
                equilibrium_forcing=False,
                reasoning="Default Stage 4 test configuration.",
            )
            assert isinstance(feedback, str)
            if feedback.startswith("VALIDATION ERRORS:"):
                raise AssertionError(feedback)
            return ""
        if isinstance(block_id, str) and block_id.startswith("observation:manifest_mean_"):
            parameter = block_id.removeprefix("observation:")
            submit_tool = next(tool for tool in tools if tool.name == "submit_prior_block")
            feedback = await submit_tool(
                priors={
                    parameter: {
                        "parameter": parameter,
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 1.0},
                        "sources": [],
                        "reasoning": "Default observation-intercept prior for scripted Stage 4 tests.",
                    }
                }
            )
            assert isinstance(feedback, str)
            if feedback.startswith("VALIDATION ERRORS:"):
                raise AssertionError(feedback)
            return ""
        if turn_index >= len(submissions):
            raise AssertionError(
                f"Unexpected extra Stage 4 turn at {block_id!r}; visited={visited_blocks}"
            )
        submission = submissions[turn_index]
        turn_index += 1
        visited_blocks.append(block_id or str(submission["block_id"]))
        visible_tools.append([tool.name for tool in tools])
        submit_tool_name = _stage4_submit_tool_name(str(submission["block_kind"]))
        submit_tool = next(tool for tool in tools if tool.name == submit_tool_name)
        feedback = await submit_tool(**_stage4_submit_tool_args(submission))
        assert isinstance(feedback, str)
        if feedback.startswith("VALIDATION ERRORS:"):
            raise AssertionError(feedback)
        return ""

    return _generate


def _make_scripted_stage4_generate_by_block(
    submissions_by_block: dict[str, dict[str, object]],
    *,
    visited_blocks: list[str],
    visible_tools: list[list[str]],
):
    """Drive ``run_stage4()`` with block-keyed scripted submissions."""

    async def _generate(messages, tools, rewrite_messages=None, rewrite_tools=None, label=None):
        del messages, rewrite_messages, rewrite_tools
        assert label is not None
        assert label.startswith("stage-4:")
        block_id = label.removeprefix("stage-4:")
        if block_id == "model:configuration":
            submit_tool = next(tool for tool in tools if tool.name == "submit_model_configuration")
            feedback = await submit_tool(
                initialization_policy="stationary",
                observation_intercept_policy="free",
                equilibrium_forcing=False,
                reasoning="Default Stage 4 test configuration.",
            )
            assert isinstance(feedback, str)
            if feedback.startswith("VALIDATION ERRORS:"):
                raise AssertionError(f"{block_id}: {feedback}")
            return ""
        if block_id.startswith("observation:manifest_mean_"):
            parameter = block_id.removeprefix("observation:")
            submit_tool = next(tool for tool in tools if tool.name == "submit_prior_block")
            feedback = await submit_tool(
                priors={
                    parameter: {
                        "parameter": parameter,
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 1.0},
                        "sources": [],
                        "reasoning": "Default observation-intercept prior for scripted Stage 4 tests.",
                    }
                }
            )
            assert isinstance(feedback, str)
            if feedback.startswith("VALIDATION ERRORS:"):
                raise AssertionError(f"{block_id}: {feedback}")
            return ""
        submission = submissions_by_block[block_id]
        visited_blocks.append(block_id)
        visible_tools.append([tool.name for tool in tools])
        submit_tool_name = _stage4_submit_tool_name(str(submission["block_kind"]))
        submit_tool = next(tool for tool in tools if tool.name == submit_tool_name)
        feedback = await submit_tool(**_stage4_submit_tool_args(submission))
        assert isinstance(feedback, str)
        if feedback.startswith("VALIDATION ERRORS:"):
            raise AssertionError(f"{block_id}: {feedback}")
        return ""

    return _generate
