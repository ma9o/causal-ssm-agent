"""Resuming a lost EpisodeWorkflow from the durable journal.

When Temporal loses a workflow's in-memory history (dev-server restart), the
artifacts survive on disk but the workflow's version-pointer state does not.
The facade reseeds a fresh workflow from the journal's ``latest_state`` /
``latest_seq`` so it resumes with stages already produced and continues its
sequence numbering, instead of re-running from ingestion.
"""

import dataclasses
import json
import uuid
from pathlib import Path

import pytest

from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.moves import RunArtifact, TransitionEffects, WriteArtifact
from nof1_causal_lab.machine.store import EpisodeJournal, TransitionRecord
from nof1_causal_lab.machine.temporal import latent_structure_activities
from nof1_causal_lab.machine.temporal.messages import EpisodeInit, MoveRequest
from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

pytestmark = pytest.mark.timeout(240)


def _valid_latent_structure() -> dict:
    return {
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


@pytest.fixture
def resume_env(monkeypatch, tmp_path):
    import nof1_causal_lab.utils.openrouter_client as openrouter_client
    from nof1_causal_lab.utils import config as config_module
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils.config import LLMProfileConfig

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    workspace_id = f"ws-{uuid.uuid4().hex[:8]}"
    input_root = Path(data_module.input_dir(workspace_id))
    input_root.mkdir(parents=True, exist_ok=True)
    (input_root / "observations.csv").write_text(
        "timestamp,steps\n2026-01-01T08:00:00,1000\n2026-01-02T08:00:00,2000\n"
    )

    config = config_module.get_config()
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda: dataclasses.replace(
            config,
            ingestion=dataclasses.replace(
                config.ingestion,
                llm=LLMProfileConfig(harness="none", model="openrouter/mock-raw"),
            ),
            structure_proposal=dataclasses.replace(
                config.structure_proposal,
                llm=LLMProfileConfig(harness="none", model="openrouter/mock-latent"),
            ),
        ),
    )

    def complete_without_derivations(store, state, produced, retracted=None):
        del store, state
        return TransitionEffects(produced=produced, retracted=retracted or [])

    monkeypatch.setattr(
        latent_structure_activities,
        "complete_derivation_cascade",
        complete_without_derivations,
    )

    async def fake_call_model(model_name, messages, tools=None, config=None, log_label=None):
        del config, log_label
        tool_names = {tool.name for tool in tools or []}
        if "execute_python" in tool_names:
            if not any(
                message.get("role") == "tool" and message.get("name") == "execute_python"
                for message in messages
            ):
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-python",
                                "type": "function",
                                "function": {
                                    "name": "execute_python",
                                    "arguments": json.dumps(
                                        {
                                            "code": (
                                                "result_df = pl.read_csv(Path(DATA_DIR) / "
                                                "'observations.csv')\n"
                                                "result_df = result_df.with_columns("
                                                "pl.col('timestamp').str.strptime(pl.Datetime))"
                                            )
                                        }
                                    ),
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
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-submit-table",
                            "type": "function",
                            "function": {
                                "name": "submit_table",
                                "arguments": json.dumps(
                                    {
                                        "column_descriptions_json": json.dumps(
                                            {
                                                "timestamp": "observation time",
                                                "steps": "step count",
                                            }
                                        )
                                    }
                                ),
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
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-latent",
                        "type": "function",
                        "function": {
                            "name": "validate_latent_structure",
                            "arguments": json.dumps(
                                {"structure_json": json.dumps(_valid_latent_structure())}
                            ),
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
    return workspace_id


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
        from nof1_causal_lab.machine.temporal.worker import build_openrouter_worker, build_worker

        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                build_worker(env.client, task_queue="test-episodes"),
                build_openrouter_worker(env.client),
            ):

                async def propose(handle, move, **kwargs):
                    return await handle.execute_update(
                        EpisodeWorkflow.propose, MoveRequest(move=move, **kwargs)
                    )

                # Run 1: drive through ingestion, then lose the workflow.
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
