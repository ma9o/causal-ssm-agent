"""Resuming a lost EpisodeWorkflow from the durable journal.

When Temporal loses a workflow's in-memory history (dev-server restart), the
artifacts survive on disk but the workflow's version-pointer state does not.
The facade reseeds a fresh workflow from the journal's ``latest_state`` /
``latest_seq`` so it resumes with stages already produced and continues its
sequence numbering, instead of re-running from stage-0.
"""

import uuid

import pytest

from nof1_causal_lab.machine import runners as runners_module
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.moves import RunArtifact, WriteArtifact
from nof1_causal_lab.machine.store import EpisodeJournal, TransitionRecord
from nof1_causal_lab.machine.temporal.messages import EpisodeInit, MoveRequest
from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

pytestmark = pytest.mark.timeout(240)


def _fake_runner(*artifact_specs):
    async def _run(workspace_id, store, pins, options):
        del workspace_id, options
        return [
            store.write_version(
                artifact_id,
                provenance="computed",
                derived_from=pins,
                produced_by=produced_by,
                json_files={f"{artifact_id}.json": {"stub": True}},
            )
            for artifact_id, produced_by in artifact_specs
        ]

    return _run


@pytest.fixture
def resume_env(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    monkeypatch.setitem(
        runners_module._TRANSITION_RUNNERS, "raw_data", _fake_runner(("raw_data", "stage-0"))
    )
    monkeypatch.setitem(
        runners_module._TRANSITION_RUNNERS,
        "latent_structure",
        _fake_runner(("latent_structure", "stage-1a")),
    )
    return f"ws-{uuid.uuid4().hex[:8]}"


def test_latest_seq_reads_max_journal_entry(resume_env):
    """``latest_seq`` returns the highest recorded seq without opening records."""
    workspace_id = resume_env
    journal = EpisodeJournal(workspace_id)
    assert journal.latest_seq() == 0  # empty journal

    for seq in (1, 2, 3):
        journal.append(
            TransitionRecord(
                seq=seq,
                ts="2026-01-01T00:00:00Z",
                move=WriteArtifact(artifact_id="question"),
                status="applied",
                state_after=EpisodeState(),
            )
        )
    assert journal.latest_seq() == 3


def test_workflow_resumes_from_seeded_init(resume_env):
    """A workflow reseeded from the journal resumes mid-chain and keeps numbering."""
    workspace_id = resume_env

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from nof1_causal_lab.machine.temporal.client import pydantic_data_converter
        from nof1_causal_lab.machine.temporal.worker import build_worker

        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with build_worker(env.client, task_queue="test-episodes"):

                async def propose(handle, move, **kwargs):
                    return await handle.execute_update(
                        EpisodeWorkflow.propose, MoveRequest(move=move, **kwargs)
                    )

                # Run 1: drive through stage-0, then lose the workflow.
                first = await env.client.start_workflow(
                    EpisodeWorkflow.run,
                    EpisodeInit(workspace_id=workspace_id),
                    id=f"episode-{workspace_id}",
                    task_queue="test-episodes",
                )
                await propose(first, WriteArtifact(artifact_id="question"), payload={"text": "q?"})
                stage0 = await propose(first, RunArtifact(artifact_id="raw_data"))
                assert stage0.status == "applied"
                await first.terminate()

                # The journal is the only surviving record of what happened.
                journal = EpisodeJournal(workspace_id)
                seed_state = journal.latest_state()
                seed_seq = journal.latest_seq()
                assert seed_state.has("question")
                assert seed_state.has("raw_data")
                assert seed_seq == stage0.seq

                # Run 2: fresh workflow, same id, seeded from the journal.
                resumed = await env.client.start_workflow(
                    EpisodeWorkflow.run,
                    EpisodeInit(
                        workspace_id=workspace_id,
                        initial_state=seed_state,
                        initial_seq=seed_seq,
                    ),
                    id=f"episode-{workspace_id}",
                    task_queue="test-episodes",
                )
                status = await resumed.query(EpisodeWorkflow.get_status)
                assert status.seq == seed_seq
                present = {a.artifact_id for a in status.artifacts if a.exists}
                assert {"question", "raw_data"} <= present

                # Downstream continues from the rehydrated state, numbering onward.
                stage1a = await propose(resumed, RunArtifact(artifact_id="latent_structure"))
                assert stage1a.status == "applied"
                assert stage1a.seq == seed_seq + 1
                assert stage1a.state.has("latent_structure")
        finally:
            await env.shutdown()

    import asyncio

    asyncio.run(scenario())
