"""EpisodeWorkflow end-to-end against a local Temporal dev server.

Covers the durable-shell contract: moves serialize through the propose
update, rejections and typed failures are journaled (not just applied
moves), state only changes on applied transitions, and staleness follows
provenance after an upstream rewrite. Stage runners are stubbed — the
real ones are exercised in their own suites; here we test the machine.
"""

import asyncio
import uuid

import pytest

from nof1_causal_lab.machine import runners as runners_module
from nof1_causal_lab.machine.errors import ModelCompileError
from nof1_causal_lab.machine.moves import RunStage, WriteArtifact
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal
from nof1_causal_lab.machine.temporal.messages import EpisodeInit, MoveRequest
from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

pytestmark = pytest.mark.timeout(240)


def _fake_runner(*artifact_specs):
    """Build a stub stage runner producing the given (artifact, produced_by) specs."""

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


async def _failing_runner(workspace_id, store, pins, options):
    del workspace_id, store, pins, options
    raise ModelCompileError(
        "stub compile failure",
        stage_id="stage-4",
        diagnostics={"hint": "prior scale unidentifiable"},
    )


@pytest.fixture
def machine_env(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    monkeypatch.setitem(
        runners_module._STAGE_RUNNERS, "stage-0", _fake_runner(("raw_data", "stage-0"))
    )
    monkeypatch.setitem(
        runners_module._STAGE_RUNNERS, "stage-1a", _fake_runner(("latent_structure", "stage-1a"))
    )
    monkeypatch.setitem(
        runners_module._STAGE_RUNNERS,
        "stage-1b",
        _fake_runner(
            ("causal_design", "stage-1b"),
            ("identification_report", "stage-1b"),
        ),
    )
    monkeypatch.setitem(
        runners_module._STAGE_RUNNERS,
        "stage-2",
        _fake_runner(("extraction_report", "stage-2"), ("model_data", "stage-2")),
    )
    monkeypatch.setitem(
        runners_module._STAGE_RUNNERS, "stage-3", _fake_runner(("validation_report", "stage-3"))
    )
    monkeypatch.setitem(runners_module._STAGE_RUNNERS, "stage-4", _failing_runner)
    return f"ws-{uuid.uuid4().hex[:8]}"


def test_episode_workflow_journey(machine_env):
    workspace_id = machine_env

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from nof1_causal_lab.machine.temporal.client import pydantic_data_converter
        from nof1_causal_lab.machine.temporal.worker import build_worker

        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with build_worker(env.client, task_queue="test-episodes"):
                handle = await env.client.start_workflow(
                    EpisodeWorkflow.run,
                    EpisodeInit(workspace_id=workspace_id),
                    id=f"episode-{workspace_id}",
                    task_queue="test-episodes",
                )

                async def propose(move, **kwargs):
                    return await handle.execute_update(
                        EpisodeWorkflow.propose, MoveRequest(move=move, **kwargs)
                    )

                # 1. Illegal move first: rejected AND journaled.
                rejected = await propose(RunStage(stage_id="stage-1b"))
                assert rejected.status == "rejected"
                assert "question" in rejected.reason

                # 2. Root write enables downstream.
                applied = await propose(
                    WriteArtifact(artifact_id="question"),
                    payload={"text": "does exercise improve sleep?"},
                )
                assert applied.status == "applied"
                assert applied.state.has("question")

                # 3. Free navigation through enabled stages (stubs).
                for stage in ("stage-0", "stage-1a", "stage-1b", "stage-2", "stage-3"):
                    outcome = await propose(RunStage(stage_id=stage))
                    assert outcome.status == "applied", (stage, outcome)

                # 4. Typed stage failure: raised, state unchanged.
                before = (await handle.query(EpisodeWorkflow.get_state)).current
                raised = await propose(RunStage(stage_id="stage-4"))
                assert raised.status == "raised"
                assert raised.error_type == "ModelCompileError"
                assert raised.diagnostics == {"hint": "prior scale unidentifiable"}
                after = (await handle.query(EpisodeWorkflow.get_state)).current
                assert after == before

                # 5. Rewriting the question stales the whole derived chain.
                status = await handle.query(EpisodeWorkflow.get_status)
                stale_before = {a.artifact_id for a in status.artifacts if a.stale}
                assert stale_before == set()
                await propose(
                    WriteArtifact(artifact_id="question"),
                    payload={"text": "does caffeine harm sleep?"},
                )
                status = await handle.query(EpisodeWorkflow.get_status)
                stale = {a.artifact_id for a in status.artifacts if a.stale}
                assert "latent_structure" in stale
                assert "causal_design" in stale
                assert "raw_data" not in stale  # not derived from question

                # 6. Journal recorded every attempt, including the rejection
                #    and the typed failure.
                records = EpisodeJournal(workspace_id).read_all()
                statuses = [record.status for record in records]
                assert statuses == [
                    "rejected",
                    "applied",  # question
                    "applied",  # stage-0
                    "applied",  # stage-1a
                    "applied",  # stage-1b
                    "applied",  # stage-2
                    "applied",  # stage-3
                    "raised",  # stage-4
                    "applied",  # question rewrite
                ]
                assert records[-2].error_type == "ModelCompileError"
                assert records[-2].diagnostics["hint"] == "prior scale unidentifiable"

                # Store kept both question versions (append-only).
                assert ArtifactStore(workspace_id).list_versions("question") == [1, 2]
        finally:
            await env.shutdown()

    asyncio.run(scenario())
