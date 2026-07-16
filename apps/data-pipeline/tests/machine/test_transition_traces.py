"""Journal-owned durable LLM traces."""

import asyncio

import pytest

from nof1_causal_lab.machine.moves import RunArtifact
from nof1_causal_lab.machine.store import (
    EpisodeJournal,
    episode_trace_path,
    promote_run_traces,
    read_episode_trace,
)
from nof1_causal_lab.machine.temporal.activities import journal_activity
from nof1_causal_lab.machine.temporal.messages import JournalInput
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage, TraceUsage


@pytest.fixture
def data_root(monkeypatch, tmp_path):
    root = tmp_path / "data"
    monkeypatch.setattr(data_module, "_DATA_URI", str(root))
    return root


def _trace(model: str = "openrouter/test-model") -> LLMTrace:
    return LLMTrace(
        messages=[TraceMessage(role="assistant", content="Done.")],
        model=model,
        usage=TraceUsage(input_tokens=3, output_tokens=5),
        total_time_seconds=0.25,
    )


def _scratch_trace(workspace_id: str, seq: int, subroutine_id: str) -> str:
    path = storage.join(
        data_module.scratch_run_dir(workspace_id, f"seq-{seq:06d}"),
        "llm",
        subroutine_id,
        "trace.json",
    )
    storage.write_text(path, _trace().model_dump_json())
    return path


def test_promotes_every_finalized_run_trace_in_subroutine_order(data_root):
    del data_root
    _scratch_trace("ws-trace", 1, "zeta")
    _scratch_trace("ws-trace", 1, "alpha")
    unfinished = storage.join(
        data_module.scratch_run_dir("ws-trace", "seq-000001"),
        "llm",
        "unfinished",
        "conversation",
        "turn.json",
    )
    storage.write_text(unfinished, "{}")

    trace_ids = promote_run_traces("ws-trace", 1)

    assert trace_ids == ["alpha", "zeta"]
    assert read_episode_trace("ws-trace", 1, "alpha")["model"] == ("openrouter/test-model")


def test_identical_promotion_is_idempotent_but_collision_is_rejected(data_root):
    del data_root
    source = _scratch_trace("ws-trace", 1, "latent-structure")
    promote_run_traces("ws-trace", 1)
    promote_run_traces("ws-trace", 1)

    storage.write_text(source, _trace("openrouter/different").model_dump_json())
    with pytest.raises(ValueError, match="trace collision"):
        promote_run_traces("ws-trace", 1)


def test_trace_id_cannot_escape_ledger_directory(data_root):
    del data_root
    with pytest.raises(ValueError, match="Invalid transition trace subroutine id"):
        episode_trace_path("ws-trace", 1, "../scratch")


def test_raised_transition_discovers_trace_and_retry_no_longer_needs_scratch(data_root):
    del data_root
    _scratch_trace("ws-trace", 1, "latent-structure")
    input = JournalInput(
        workspace_id="ws-trace",
        seq=1,
        move=RunArtifact(artifact_id="latent_structure"),
        status="raised",
        error_type="LLMSubroutineError",
        error_message="validation failed",
        resume=None,
    )

    asyncio.run(journal_activity(input))
    storage.rm_tree(data_module.scratch_run_dir("ws-trace", "seq-000001"))
    asyncio.run(journal_activity(input))

    record = EpisodeJournal("ws-trace").read(1)
    assert record is not None
    assert record.status == "raised"
    assert record.trace_ids == ["latent-structure"]
    assert read_episode_trace("ws-trace", 1, "latent-structure")["model"] == (
        "openrouter/test-model"
    )
