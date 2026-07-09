"""Canonical LLM trace storage refs."""

import pytest

from nof1_causal_lab.machine.trace_store import (
    FILE_TRACE_PREFIX,
    FileTraceStore,
    TraceMetadata,
    read_trace,
)
from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage, TraceUsage


@pytest.fixture
def data_root(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    root = tmp_path / "data"
    monkeypatch.setattr(data_module, "DATA_URI", str(root))
    return root


def _trace() -> LLMTrace:
    return LLMTrace(
        messages=[TraceMessage(role="assistant", content="Done.")],
        model="openrouter/test-model",
        usage=TraceUsage(input_tokens=3, output_tokens=5),
        total_time_seconds=0.25,
    )


def _metadata(workspace_id: str = "ws-trace") -> TraceMetadata:
    return TraceMetadata(
        workspace_id=workspace_id,
        run_id="seq-000001",
        subroutine_id="latent-structure",
        context_kind="latent_structure",
    )


def test_file_trace_store_writes_data_relative_ref(data_root):
    store = FileTraceStore()

    ref = store.write_trace(_trace(), _metadata())

    assert ref.startswith(FILE_TRACE_PREFIX + "ws-trace/")
    loaded = read_trace("ws-trace", ref)
    assert loaded.model == "openrouter/test-model"
    assert loaded.messages[0].content == "Done."
    assert loaded.usage.input_tokens == 3
    assert not str(ref).startswith(str(data_root))


def test_file_trace_store_rejects_cross_workspace_ref(data_root):
    store = FileTraceStore()
    ref = store.write_trace(_trace(), _metadata("ws-a"))

    with pytest.raises(ValueError, match="does not belong"):
        read_trace("ws-b", ref)


def test_file_trace_store_rejects_parent_traversal(data_root):
    with pytest.raises(ValueError, match="Invalid file trace ref path"):
        read_trace("ws-trace", FILE_TRACE_PREFIX + "ws-trace/../trace.json")
