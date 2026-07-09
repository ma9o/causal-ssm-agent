import json
import uuid

import polars as pl
import pytest

from nof1_causal_lab.machine.temporal.llm_subroutine_activities import (
    append_llm_repair_message_activity,
    execute_llm_tool_calls_activity,
)
from nof1_causal_lab.machine.temporal.llm_subroutine_workflow import LLMSubroutineWorkflow
from nof1_causal_lab.machine.temporal.measurement_activities import (
    call_openrouter_activity,
)
from nof1_causal_lab.machine.temporal.measurement_workflow import ExtractionChunkWorkflow
from nof1_causal_lab.machine.temporal.messages import (
    AppendLLMRepairMessageInput,
    ExtractionChunkWorkflowInput,
    LLMBackendConfig,
    LLMSubroutineInput,
    LLMToolExecutionInput,
    LLMToolSpec,
    OpenRouterCallInput,
    OpenRouterLLMConfig,
)
from nof1_causal_lab.utils import storage
from tests.helpers import run_async

pytestmark = pytest.mark.timeout(240)


def test_call_openrouter_activity_reuses_persisted_call_result(monkeypatch, tmp_path):
    import nof1_causal_lab.utils.openrouter_client as openrouter_client

    calls: list[list[dict]] = []
    conversation_ref = str(tmp_path / "conversation.json")
    next_conversation_ref = str(tmp_path / "conversation-next.json")
    call_ref = str(tmp_path / "call.json")
    assistant_ref = str(tmp_path / "assistant.json")
    storage.write_text(
        conversation_ref,
        json.dumps({"messages": [{"role": "user", "content": "extract"}]}),
    )

    async def fake_call_model(model_name, messages, tools=None, config=None, log_label=None):
        del tools, config, log_label
        calls.append(messages)
        return {
            "message": {"role": "assistant", "content": "", "tool_calls": []},
            "completion": "",
            "usage": {"input_tokens": 3, "output_tokens": 5, "reasoning_tokens": None},
            "model": model_name,
            "time": 0.25,
            "stop_reason": "tool_calls",
        }

    monkeypatch.setattr(openrouter_client, "call_model", fake_call_model)

    input = OpenRouterCallInput(
        conversation_ref=conversation_ref,
        next_conversation_ref=next_conversation_ref,
        call_ref=call_ref,
        assistant_ref=assistant_ref,
        llm=OpenRouterLLMConfig(model="openrouter/mock", timeout=120),
        tools=[
            LLMToolSpec(
                name="validate_extractions",
                description="Validate output.",
                param_name="output_json",
                param_description="Worker output JSON.",
            )
        ],
        log_label="test",
    )

    first = run_async(call_openrouter_activity(input))
    second = run_async(call_openrouter_activity(input))

    assert len(calls) == 1
    assert first == second
    assert first.model == "openrouter/mock"
    assert storage.exists(call_ref)


def test_execute_llm_tool_calls_activity_dispatches_by_tool_name(tmp_path):
    context_ref = str(tmp_path / "context.json")
    assistant_ref = str(tmp_path / "assistant.json")
    conversation_ref = str(tmp_path / "conversation.json")
    execution_ref = str(tmp_path / "tool-execution.json")
    result_ref = str(tmp_path / "result.json")

    output_json = json.dumps(
        {
            "extractions": [
                {
                    "window_start": "2026-01-01T00:00:00",
                    "indicator": "steps",
                    "value": 1000,
                }
            ]
        }
    )
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "submit_extractions",
            "arguments": json.dumps({"output_json": output_json}),
        },
    }
    storage.write_text(
        context_ref,
        json.dumps(
            {
                "question": "does exercise improve sleep?",
                "window_text": "2026-01-01T00:00:00: steps were 1000",
                "window_starts": ["2026-01-01T00:00:00"],
                "measurement_structure": {
                    "indicators": [
                        {
                            "name": "steps",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "support_kind": "interval",
                            "summary_operator": "mean",
                            "anchor_policy": "window_end",
                        }
                    ]
                },
            }
        ),
    )
    storage.write_text(
        assistant_ref,
        json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}),
    )
    storage.write_text(
        conversation_ref,
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "extract"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ]
            }
        ),
    )

    result = run_async(
        execute_llm_tool_calls_activity(
            LLMToolExecutionInput(
                workspace_id="ws-test",
                run_id="seq-000001",
                subroutine_id="measurement-extraction",
                context_kind="measurement_extraction",
                context_ref=context_ref,
                conversation_ref=conversation_ref,
                assistant_ref=assistant_ref,
                execution_ref=execution_ref,
                result_ref=result_ref,
                tools=[
                    LLMToolSpec(
                        name="validate_extractions",
                        description="Validate worker extraction output JSON.",
                        param_name="output_json",
                        param_description="The JSON string containing the worker output.",
                    ),
                    LLMToolSpec(
                        name="submit_extractions",
                        description="Submit worker extraction output JSON.",
                        param_name="output_json",
                        param_description="The JSON string containing the worker output.",
                    ),
                ],
            )
        )
    )

    assert result.terminal_success is True
    assert result.result_ref == result_ref
    assert result.tool_calls_fired == ["submit_extractions"]
    assert storage.read_json(result_ref)["extractions"][0]["value"] == 1000


def test_execute_llm_tool_calls_activity_persists_raw_data_submit_table(tmp_path):
    context_ref = str(tmp_path / "context.json")
    assistant_ref = str(tmp_path / "assistant.json")
    conversation_ref = str(tmp_path / "conversation.json")
    execution_ref = str(tmp_path / "tool-execution.json")
    result_ref = str(tmp_path / "result.json")
    dataframe_ref = str(tmp_path / "latest-dataframe.ipc")

    dataframe = pl.DataFrame(
        {
            "timestamp": ["2026-01-01T08:00:00", "2026-01-02T08:00:00"],
            "steps": [1000, 2000],
        }
    ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime))
    with storage.open_file(dataframe_ref, "wb") as file:
        dataframe.write_ipc(file)

    output_json = json.dumps({"timestamp": "observation time", "steps": "step count"})
    tool_call = {
        "id": "call-raw-submit",
        "type": "function",
        "function": {
            "name": "submit_table",
            "arguments": json.dumps({"column_descriptions_json": output_json}),
        },
    }
    storage.write_text(
        context_ref,
        json.dumps({"extract_dir": str(tmp_path), "dataframe_ref": dataframe_ref}),
    )
    storage.write_text(
        assistant_ref,
        json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}),
    )
    storage.write_text(
        conversation_ref,
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "ingest"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ]
            }
        ),
    )

    result = run_async(
        execute_llm_tool_calls_activity(
            LLMToolExecutionInput(
                workspace_id="ws-test",
                run_id="seq-000001",
                subroutine_id="raw-data",
                context_kind="raw_data_ingestion",
                context_ref=context_ref,
                conversation_ref=conversation_ref,
                assistant_ref=assistant_ref,
                execution_ref=execution_ref,
                result_ref=result_ref,
                tools=[
                    LLMToolSpec(
                        name="submit_table",
                        description="Validate and finalize the ingested DataFrame.",
                        param_name="column_descriptions_json",
                        param_description="JSON object mapping column names to descriptions.",
                        executor="raw_data_submit_table",
                    )
                ],
            )
        )
    )

    assert result.terminal_success is True
    assert result.result_ref == result_ref
    persisted = storage.read_json(result_ref)
    assert persisted["dataframe_ref"] == dataframe_ref
    assert persisted["column_descriptions"]["steps"] == "step count"


def test_execute_llm_tool_calls_activity_executes_raw_python_locally(tmp_path):
    context_ref = str(tmp_path / "context.json")
    assistant_ref = str(tmp_path / "assistant.json")
    conversation_ref = str(tmp_path / "conversation.json")
    execution_ref = str(tmp_path / "tool-execution.json")
    result_ref = str(tmp_path / "result.json")
    dataframe_ref = str(tmp_path / "latest-dataframe.ipc")
    (tmp_path / "observations.csv").write_text("timestamp,steps\n2026-01-01T08:00:00,1000\n")

    tool_call = {
        "id": "call-python",
        "type": "function",
        "function": {
            "name": "execute_python",
            "arguments": json.dumps(
                {
                    "code": (
                        "result_df = pl.read_csv(Path(DATA_DIR) / 'observations.csv')\n"
                        "result_df = result_df.with_columns("
                        "pl.col('timestamp').str.strptime(pl.Datetime))"
                    )
                }
            ),
        },
    }
    storage.write_text(
        context_ref,
        json.dumps({"extract_dir": str(tmp_path), "dataframe_ref": dataframe_ref}),
    )
    storage.write_text(
        assistant_ref,
        json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}),
    )
    storage.write_text(
        conversation_ref,
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "ingest"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ]
            }
        ),
    )

    result = run_async(
        execute_llm_tool_calls_activity(
            LLMToolExecutionInput(
                workspace_id="ws-test",
                run_id="seq-000001",
                subroutine_id="raw-data",
                context_kind="raw_data_ingestion",
                context_ref=context_ref,
                conversation_ref=conversation_ref,
                assistant_ref=assistant_ref,
                execution_ref=execution_ref,
                result_ref=result_ref,
                tools=[
                    LLMToolSpec(
                        name="execute_python",
                        description="Execute Python.",
                        kind="checkpoint",
                        executor="raw_data_execute_python",
                        parameters_schema={
                            "type": "object",
                            "properties": {"code": {"type": "string"}},
                            "required": ["code"],
                            "additionalProperties": False,
                        },
                    )
                ],
            )
        )
    )

    assert result.terminal_success is False
    assert "Success!" in result.feedback_preview
    with storage.open_file(dataframe_ref, "rb") as file:
        dataframe = pl.read_ipc(file)
    assert dataframe["steps"].to_list() == [1000]


def test_execute_llm_tool_calls_activity_returns_recoverable_tool_exception(tmp_path):
    context_ref = str(tmp_path / "context.json")
    assistant_ref = str(tmp_path / "assistant.json")
    conversation_ref = str(tmp_path / "conversation.json")
    execution_ref = str(tmp_path / "tool-execution.json")
    result_ref = str(tmp_path / "result.json")
    tool_call = {
        "id": "call-submit",
        "type": "function",
        "function": {
            "name": "submit_table",
            "arguments": json.dumps({"column_descriptions_json": "{}"}),
        },
    }
    storage.write_text(context_ref, json.dumps({"extract_dir": str(tmp_path)}))
    storage.write_text(
        assistant_ref,
        json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}),
    )
    storage.write_text(
        conversation_ref,
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "ingest"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ]
            }
        ),
    )

    result = run_async(
        execute_llm_tool_calls_activity(
            LLMToolExecutionInput(
                workspace_id="ws-test",
                run_id="seq-000001",
                subroutine_id="raw-data",
                context_kind="raw_data_ingestion",
                context_ref=context_ref,
                conversation_ref=conversation_ref,
                assistant_ref=assistant_ref,
                execution_ref=execution_ref,
                result_ref=result_ref,
                tools=[
                    LLMToolSpec(
                        name="submit_table",
                        description="Submit table.",
                        executor="raw_data_submit_table",
                    )
                ],
            )
        )
    )

    assert result.terminal_success is False
    assert result.result_ref is None
    assert result.feedback_preview.startswith("Tool execution failed:")


def test_append_llm_repair_message_activity_persists_repair_turn(tmp_path):
    conversation_ref = str(tmp_path / "conversation.json")
    next_conversation_ref = str(tmp_path / "conversation-repair.json")
    storage.write_text(
        conversation_ref,
        json.dumps({"messages": [{"role": "user", "content": "try tool"}]}),
    )

    result = run_async(
        append_llm_repair_message_activity(
            AppendLLMRepairMessageInput(
                workspace_id="ws-test",
                run_id="seq-000001",
                subroutine_id="repair",
                conversation_ref=conversation_ref,
                next_conversation_ref=next_conversation_ref,
                error_text="provider rejected malformed tool context",
                tools=[
                    LLMToolSpec(
                        name="submit_table",
                        description="Submit table.",
                        executor="raw_data_submit_table",
                    )
                ],
            )
        )
    )

    assert result.conversation_ref == next_conversation_ref
    messages = storage.read_json(next_conversation_ref)["messages"]
    assert messages[-1]["role"] == "user"
    assert "Your previous response could not be processed" in messages[-1]["content"]
    assert "submit_table" in messages[-1]["content"]


def test_execute_llm_tool_calls_activity_terminal_without_result_ref(tmp_path):
    context_ref = str(tmp_path / "context.json")
    assistant_ref = str(tmp_path / "assistant.json")
    conversation_ref = str(tmp_path / "conversation.json")
    execution_ref = str(tmp_path / "tool-execution.json")
    result_ref = str(tmp_path / "result.json")
    (tmp_path / "observations.csv").write_text("timestamp,steps\n2026-01-01T08:00:00,1000\n")

    tool_call = {
        "id": "call-list",
        "type": "function",
        "function": {
            "name": "list_files",
            "arguments": json.dumps({"path": "."}),
        },
    }
    storage.write_text(
        context_ref,
        json.dumps(
            {
                "extract_dir": str(tmp_path),
                "dataframe_ref": str(tmp_path / "latest-dataframe.ipc"),
            }
        ),
    )
    storage.write_text(
        assistant_ref,
        json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}),
    )
    storage.write_text(
        conversation_ref,
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "ingest"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ]
            }
        ),
    )

    result = run_async(
        execute_llm_tool_calls_activity(
            LLMToolExecutionInput(
                workspace_id="ws-test",
                run_id="seq-000001",
                subroutine_id="raw-data",
                context_kind="raw_data_ingestion",
                context_ref=context_ref,
                conversation_ref=conversation_ref,
                assistant_ref=assistant_ref,
                execution_ref=execution_ref,
                result_ref=result_ref,
                tools=[
                    LLMToolSpec(
                        name="list_files",
                        description="List files.",
                        kind="terminal",
                        executor="raw_data_list_files",
                        success_output=None,
                        parameters_schema={
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": [],
                            "additionalProperties": False,
                        },
                    )
                ],
            )
        )
    )

    assert result.terminal_success is True
    assert result.result_ref is None
    assert storage.exists(result_ref) is False


def test_extraction_chunk_workflow_runs_shared_llm_subroutine(monkeypatch, tmp_path):
    from temporalio.testing import WorkflowEnvironment

    import nof1_causal_lab.utils.openrouter_client as openrouter_client
    from nof1_causal_lab.machine.temporal.client import pydantic_data_converter
    from nof1_causal_lab.machine.temporal.worker import build_openrouter_worker, build_worker
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    output_json = json.dumps(
        {
            "extractions": [
                {
                    "window_start": "2026-01-01T00:00:00",
                    "indicator": "steps",
                    "value": 1000,
                }
            ]
        }
    )

    async def fake_call_model(model_name, messages, tools=None, config=None, log_label=None):
        del messages, tools, config, log_label
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-extract",
                        "type": "function",
                        "function": {
                            "name": "validate_extractions",
                            "arguments": json.dumps({"output_json": output_json}),
                        },
                    }
                ],
            },
            "completion": "",
            "usage": {"input_tokens": 3, "output_tokens": 5, "reasoning_tokens": None},
            "model": model_name,
            "time": 0.25,
            "stop_reason": "tool_calls",
        }

    monkeypatch.setattr(openrouter_client, "call_model", fake_call_model)

    workspace_id = f"ws-{uuid.uuid4().hex[:8]}"
    spec_ref = str(tmp_path / "chunk.json")
    storage.write_text(
        spec_ref,
        json.dumps(
            {
                "worker_id": 0,
                "question": "does exercise improve sleep?",
                "window_text": "2026-01-01T00:00:00: steps were 1000",
                "window_starts": ["2026-01-01T00:00:00"],
                "measurement_structure": {
                    "indicators": [
                        {
                            "name": "steps",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "support_kind": "interval",
                            "summary_operator": "mean",
                            "anchor_policy": "window_end",
                        }
                    ]
                },
            }
        ),
    )

    async def scenario():
        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                build_worker(env.client, task_queue="test-episodes"),
                build_openrouter_worker(env.client),
            ):
                result = await env.client.execute_workflow(
                    ExtractionChunkWorkflow.run,
                    ExtractionChunkWorkflowInput(
                        workspace_id=workspace_id,
                        run_id="seq-000001",
                        worker_id=0,
                        n_windows=1,
                        spec_ref=spec_ref,
                        attempt=1,
                        llm=LLMBackendConfig(
                            harness="none",
                            model="openrouter/mock-extraction",
                            timeout=120,
                        ),
                        max_tool_turns=3,
                    ),
                    id=f"extract-chunk-{workspace_id}",
                    task_queue="test-episodes",
                )
        finally:
            await env.shutdown()
        return result

    result = run_async(scenario())

    assert result.status == "completed"
    assert result.n_extractions == 1
    assert result.n_llm_calls == 1
    assert result.result_ref is not None
    assert result.trace_ref is not None
    assert storage.read_json(result.result_ref)["dataframe"][0]["value"] == "1000"
    from nof1_causal_lab.machine.trace_store import read_trace

    trace = read_trace(workspace_id, result.trace_ref)
    assert trace.model == "openrouter/mock-extraction"
    assert trace.usage.input_tokens == 3


def test_llm_subroutine_workflow_runs_openrouter_without_tool(monkeypatch, tmp_path):
    from temporalio.testing import WorkflowEnvironment

    import nof1_causal_lab.utils.openrouter_client as openrouter_client
    from nof1_causal_lab.machine.temporal.client import pydantic_data_converter
    from nof1_causal_lab.machine.temporal.worker import build_openrouter_worker, build_worker
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))

    async def fake_call_model(model_name, messages, tools=None, config=None, log_label=None):
        del messages, config, log_label
        assert tools is None
        return {
            "message": {"role": "assistant", "content": "Summary text.", "tool_calls": []},
            "completion": "Summary text.",
            "usage": {"input_tokens": 7, "output_tokens": 4, "reasoning_tokens": None},
            "model": model_name,
            "time": 0.1,
            "stop_reason": "end_turn",
        }

    monkeypatch.setattr(openrouter_client, "call_model", fake_call_model)

    workspace_id = f"ws-{uuid.uuid4().hex[:8]}"
    context_ref = str(tmp_path / "analysis-context.json")
    storage.write_text(
        context_ref,
        json.dumps(
            {
                "system_prompt": "Write concise analysis commentary.",
                "user_messages": ["Summarize the result."],
            }
        ),
    )

    async def scenario():
        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                build_worker(env.client, task_queue="test-episodes"),
                build_openrouter_worker(env.client),
            ):
                result = await env.client.execute_workflow(
                    LLMSubroutineWorkflow.run,
                    LLMSubroutineInput(
                        workspace_id=workspace_id,
                        run_id="seq-000001",
                        subroutine_id="analysis-commentary",
                        context_kind="analysis_commentary",
                        context_ref=context_ref,
                        llm=LLMBackendConfig(
                            harness="none",
                            model="openrouter/mock-analysis",
                            timeout=120,
                        ),
                        max_tool_turns=1,
                        require_result=False,
                    ),
                    id=f"analysis-commentary-{workspace_id}",
                    task_queue="test-episodes",
                )
        finally:
            await env.shutdown()
        return result

    result = run_async(scenario())

    assert result.result_ref is None
    assert result.n_llm_calls == 1
    assert result.trace_ref is not None
    from nof1_causal_lab.machine.trace_store import read_trace

    trace = read_trace(workspace_id, result.trace_ref)
    assert trace.model == "openrouter/mock-analysis"
    assert trace.messages[-1].content == "Summary text."


@pytest.mark.parametrize(
    ("harness", "model"),
    [
        ("claude-code", "claude/mock-latent"),
        ("codex", "codex/mock-latent"),
    ],
)
def test_llm_subroutine_workflow_delegates_harness_tool_to_temporal_activity(
    monkeypatch, tmp_path, harness, model
):
    from contextlib import asynccontextmanager

    from temporalio.testing import WorkflowEnvironment

    import nof1_causal_lab.utils.harness.claude as claude_harness
    import nof1_causal_lab.utils.harness.codex as codex_harness
    from nof1_causal_lab.machine.temporal.client import (
        HARNESS_CLAUDE_TASK_QUEUE,
        HARNESS_CODEX_TASK_QUEUE,
        pydantic_data_converter,
    )
    from nof1_causal_lab.machine.temporal.worker import build_harness_worker, build_worker
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils.agent_session import AgentResult, TurnResult
    from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage, TraceUsage

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    valid_structure = {
        "constructs": [
            {
                "name": "exercise",
                "description": "exercise level",
                "role": "exogenous",
                "is_outcome": False,
                "temporal_status": "time_varying",
            },
            {
                "name": "sleep",
                "description": "sleep quality",
                "role": "endogenous",
                "is_outcome": True,
                "temporal_status": "time_varying",
            },
        ],
        "edges": [
            {
                "cause": "exercise",
                "effect": "sleep",
                "description": "exercise can affect sleep",
                "lagged": True,
                "sources": [],
            }
        ],
    }

    class FakeHarnessSession:
        def __init__(self, tools):
            self._tools = tools
            self.session_id = "fake-claude-session"
            self.raw_events = [{"type": "system", "subtype": "init"}]
            self._tool_output = ""

        async def turn(self, user_message):
            del user_message
            self._tool_output = await self._tools[0].execute(
                structure_json=json.dumps(valid_structure)
            )
            self.raw_events.append(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "validate_latent_structure",
                            }
                        ]
                    },
                }
            )
            self.raw_events.append(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "content": self._tool_output,
                            }
                        ]
                    },
                }
            )
            return TurnResult(
                completion="",
                terminal_tool_name="validate_latent_structure",
                terminal_tool_output=self._tool_output,
                tool_calls_fired=["validate_latent_structure"],
            )

        @property
        def result(self):
            return AgentResult(
                completion="",
                trace=LLMTrace(
                    messages=[
                        TraceMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                {
                                    "id": "fake-tool-call",
                                    "type": "function",
                                    "function": {"name": "validate_latent_structure"},
                                }
                            ],
                        ),
                        TraceMessage(
                            role="tool",
                            content=self._tool_output,
                            tool_name="validate_latent_structure",
                            tool_result=self._tool_output,
                        ),
                    ],
                    model=f"fake-{harness}",
                    total_time_seconds=0.1,
                    usage=TraceUsage(input_tokens=1, output_tokens=1),
                ),
                terminal_tool_name="validate_latent_structure",
                terminal_tool_output=self._tool_output,
            )

        async def aclose(self):
            return None

    @asynccontextmanager
    async def fake_open_claude_harness_session(**kwargs):
        yield FakeHarnessSession(kwargs["tools"])

    @asynccontextmanager
    async def fake_open_codex_harness_session(**kwargs):
        yield FakeHarnessSession(kwargs["tools"])

    monkeypatch.setattr(
        claude_harness,
        "open_claude_harness_session",
        fake_open_claude_harness_session,
    )
    monkeypatch.setattr(
        codex_harness,
        "open_codex_harness_session",
        fake_open_codex_harness_session,
    )
    harness_task_queue = (
        HARNESS_CLAUDE_TASK_QUEUE if harness == "claude-code" else HARNESS_CODEX_TASK_QUEUE
    )

    workspace_id = f"ws-{uuid.uuid4().hex[:8]}"
    context_ref = str(tmp_path / "latent-context.json")
    storage.write_text(
        context_ref,
        json.dumps(
            {
                "system_prompt": "Propose a latent structure.",
                "user_messages": ["Use the validation tool."],
            }
        ),
    )

    async def scenario():
        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        monkeypatch.setenv("TEMPORAL_ADDRESS", env.client.service_client.config.target_host)
        monkeypatch.setenv("TEMPORAL_NAMESPACE", env.client.namespace)
        try:
            async with (
                build_worker(env.client, task_queue="test-episodes"),
                build_harness_worker(
                    env.client,
                    harness_task_queue,
                    max_concurrent_activities=1,
                ),
            ):
                handle = await env.client.start_workflow(
                    LLMSubroutineWorkflow.run,
                    LLMSubroutineInput(
                        workspace_id=workspace_id,
                        run_id="seq-000001",
                        subroutine_id="latent-structure",
                        context_kind="latent_structure",
                        context_ref=context_ref,
                        llm=LLMBackendConfig(
                            harness=harness,
                            model=model,
                            timeout=10,
                        ),
                        max_tool_turns=1,
                    ),
                    id=f"harness-latent-{workspace_id}",
                    task_queue="test-episodes",
                )
                result = await handle.result()
                history = await handle.fetch_history()
                activity_names = [
                    event.activity_task_scheduled_event_attributes.activity_type.name
                    for event in history.events
                    if event.activity_task_scheduled_event_attributes.activity_type.name
                ]
        finally:
            await env.shutdown()
        return result, activity_names

    result, activity_names = run_async(scenario())

    assert result.result_ref is not None
    assert result.n_harness_turns == 1
    assert "execute_harness_tool_request_activity" in activity_names
    assert storage.read_json(result.result_ref)["latent_structure"] == valid_structure
    tool_root = storage.join(
        data_module.runs_dir(workspace_id),
        "temporal-llm",
        "seq-000001",
        "latent-structure",
        "harness-tools",
        "user-001",
    )
    requests = storage.listdir(storage.join(tool_root, "requests"))
    responses = storage.listdir(storage.join(tool_root, "responses"))
    assert len(requests) == 1
    assert len(responses) == 1
    response = storage.read_json(responses[0])
    assert response["tool_name"] == "validate_latent_structure"
    assert response["success"] is True
