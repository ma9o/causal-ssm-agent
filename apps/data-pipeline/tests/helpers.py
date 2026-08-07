"""Shared cross-cutting test helpers (LLM/session fakes, async runners)."""

import asyncio
from typing import Any


def run_async(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)


def invalid_dict_payload(value: object) -> Any:
    return value


def make_prior_plan(statistical_model_spec, priors):
    """Project test prior payloads into the strict executable compiler contract."""
    from nof1_causal_lab.artifacts.prior import ExecutablePrior
    from nof1_causal_lab.models.prior_planning import build_prior_plan

    entries = []
    for parameter, value in priors.items():
        payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
        entries.append(
            ExecutablePrior.model_validate(
                {
                    "parameter": parameter,
                    "distribution": payload["distribution"],
                    "params": payload["params"],
                    "reference_interval_days": payload.get("reference_interval_days"),
                }
            )
        )
    return build_prior_plan(statistical_model_spec, entries)


def make_structural_plan(
    state_names: list[str],
    edges: list[tuple[str, str]],
) -> dict[str, Any]:
    """Build a minimal strict StructuralPlan for topology-focused tests."""
    construct_id = {name: f"construct:{index:04d}" for index, name in enumerate(state_names)}
    edge_ids = [f"edge:{index:04d}" for index in range(len(edges))]
    indicator_id = {name: f"indicator:{index:04d}" for index, name in enumerate(state_names)}
    return {
        "schema_version": 1,
        "semantics": {
            "constructs": {
                construct_id[name]: {
                    "name": name,
                    "description": name,
                    "role": "endogenous",
                    "is_outcome": False,
                    "temporal_status": "time_varying",
                }
                for name in state_names
            },
            "edges": {
                source_id: {
                    "cause": cause,
                    "effect": effect,
                    "description": f"{cause} causes {effect}",
                    "lagged": True,
                    "sources": [],
                }
                for source_id, (cause, effect) in zip(edge_ids, edges, strict=True)
            },
            "indicators": {
                indicator_id[name]: {
                    "name": f"{name}_obs",
                    "construct_name": name,
                    "how_to_measure": f"measure {name}",
                    "construct_polarity": "positive",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "source_columns": [],
                    "extraction_mode": "semantic",
                }
                for name in state_names
            },
            "model_clock": "1d",
        },
        "state_order": [construct_id[name] for name in state_names],
        "edges": [
            {
                "source_id": source_id,
                "cause_id": construct_id[cause],
                "effect_id": construct_id[effect],
                "lagged": True,
            }
            for source_id, (cause, effect) in zip(edge_ids, edges, strict=True)
        ],
        "manifest_indicator_order": [indicator_id[name] for name in state_names],
        "reference_indicator_ids": {construct_id[name]: indicator_id[name] for name in state_names},
        "known_inputs": [],
        "induced_dependencies": [],
        "dispositions": [
            *[
                {
                    "source_id": construct_id[name],
                    "source_kind": "construct",
                    "disposition": "retained_state",
                    "reason": "test state",
                }
                for name in state_names
            ],
            *[
                {
                    "source_id": source_id,
                    "source_kind": "edge",
                    "disposition": "retained_edge",
                    "reason": "test edge",
                }
                for source_id in edge_ids
            ],
            *[
                {
                    "source_id": indicator_id[name],
                    "source_kind": "indicator",
                    "disposition": "manifest",
                    "reason": "test manifest",
                }
                for name in state_names
            ],
        ],
    }


def make_mock_session_factory(responses: list[str]):
    """Create a mock ``ScopedSessionFactory``-shaped object for tests.

    When ``.open(..., tools=[...])`` is entered and ``.turn()`` is called,
    the mock consumes the next canned response, invokes the first tool
    with it (so ``make_context_tool`` capture dicts populate the same way
    they do in the real tool loop), and returns it as the turn completion.

    """
    from contextlib import asynccontextmanager

    from nof1_causal_lab.utils.agent_session import AgentResult, TurnResult
    from nof1_causal_lab.utils.llm import LLMTrace

    call_count = [0]

    class _MockSession:
        def __init__(self, tools):
            self._tools = tools or []
            self._last_completion = ""

        async def turn(self, user_message: str) -> TurnResult:
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            response = responses[idx]

            if self._tools:
                tool = self._tools[0]
                props = tool.parameters.get("properties", {})
                required = tool.parameters.get("required", [])
                param_name = required[0] if required else next(iter(props), None)
                if param_name:
                    await tool(**{param_name: response})
                else:
                    await tool(response)

            self._last_completion = response
            return TurnResult(completion=response)

        @property
        def result(self) -> AgentResult:
            return AgentResult(completion=self._last_completion, trace=LLMTrace())

    class _MockFactory:
        @asynccontextmanager
        async def open(self, *, system_prompt=None, tools=None, log_label=None):
            yield _MockSession(tools)

    return _MockFactory()
