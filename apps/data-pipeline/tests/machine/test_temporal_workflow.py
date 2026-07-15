"""EpisodeWorkflow end-to-end against a local Temporal dev server.

Covers the durable-shell contract: moves serialize through the propose
update, rejections and typed failures are journaled (not just applied
moves), state only changes on applied transitions, and staleness follows
provenance after an upstream rewrite. Stage runners are stubbed — the
real ones are exercised in their own suites; here we test the machine.
"""

import asyncio
import dataclasses
import json
import uuid
from pathlib import Path

import pytest

from nof1_causal_lab.machine import runners as runners_module
from nof1_causal_lab.machine.moves import RunArtifact, TransitionEffects, WriteArtifact
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal
from nof1_causal_lab.machine.temporal import (
    latent_structure_activities,
    measurement_activities,
    measurement_structure_activities,
)
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


def _valid_measurement_structure() -> dict:
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "name": "sleep_steps_proxy",
                "construct_name": "sleep",
                "how_to_measure": "Use the `steps` column directly as a placeholder sleep proxy.",
                "construct_polarity": "positive",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["steps"],
                "extraction_mode": "computed",
            }
        ],
    }


@pytest.fixture
def machine_env(monkeypatch, tmp_path):
    import nof1_causal_lab.flows.transitions.model_spec.assembly as model_spec_assembly
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
            prior_elicitation=dataclasses.replace(
                config.prior_elicitation,
                llm=LLMProfileConfig(harness="none", model="openrouter/mock-model-spec"),
            ),
        ),
    )

    def fake_materialize_model_spec_result(**kwargs):
        del kwargs
        return {
            "statistical_model_spec": {"likelihoods": [], "parameters": []},
            "authored_priors": {},
        }

    monkeypatch.setattr(
        model_spec_assembly,
        "materialize_model_spec_result",
        fake_materialize_model_spec_result,
    )

    def complete_without_derivations(store, state, produced, retracted=None):
        del state
        extra = []
        measurement_structure = next(
            (info for info in produced if info.artifact_id == "measurement_structure"),
            None,
        )
        if measurement_structure is not None:
            extra.extend(
                [
                    store.write_version(
                        "causal_design",
                        provenance="computed",
                        derived_from={"measurement_structure": measurement_structure.version},
                        produced_by="derive:causal_design",
                        json_files={
                            "causal_design.json": {
                                "causal_design": {
                                    "latent": {"constructs": []},
                                    "measurement": {"indicators": []},
                                    "estimation": {"state_order": [], "edges": []},
                                }
                            }
                        },
                    ),
                    store.write_version(
                        "identification_report",
                        provenance="computed",
                        derived_from={"causal_design": 1},
                        produced_by="derive:identification_report",
                        json_files={
                            "identification_report.json": {
                                "estimable_treatments": ["sleep_steps_proxy"]
                            }
                        },
                    ),
                ]
            )
        if any(info.artifact_id == "measurements" for info in produced):
            extra.append(
                store.write_version(
                    "validation_report",
                    provenance="computed",
                    derived_from={},
                    produced_by="derive:validation_report",
                    json_files={"validation_report.json": {"indicators": {}}},
                )
            )
        return TransitionEffects(produced=[*produced, *extra], retracted=retracted or [])

    monkeypatch.setattr(
        runners_module,
        "complete_derivation_cascade",
        complete_without_derivations,
    )
    monkeypatch.setattr(
        measurement_activities,
        "complete_derivation_cascade",
        complete_without_derivations,
    )
    monkeypatch.setattr(
        latent_structure_activities,
        "complete_derivation_cascade",
        complete_without_derivations,
    )
    monkeypatch.setattr(
        measurement_structure_activities,
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
        tool_name = tools[0].name if tools else ""
        if tool_name == "validate_measurement_structure":
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-measurement",
                            "type": "function",
                            "function": {
                                "name": "validate_measurement_structure",
                                "arguments": json.dumps(
                                    {"measurement_json": json.dumps(_valid_measurement_structure())}
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


def test_episode_workflow_journey(machine_env):
    workspace_id = machine_env

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from nof1_causal_lab.machine.temporal.client import pydantic_data_converter
        from nof1_causal_lab.machine.temporal.worker import (
            build_model_spec_simulation_worker,
            build_openrouter_worker,
            build_worker,
        )

        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                build_worker(env.client, task_queue="test-episodes"),
                build_openrouter_worker(env.client),
                build_model_spec_simulation_worker(env.client),
            ):
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
                rejected = await propose(RunArtifact(artifact_id="measurement_structure"))
                assert rejected.status == "rejected"
                assert "question" in rejected.reason

                # 2. Root write enables downstream.
                applied = await propose(
                    WriteArtifact(artifact_id="question"),
                    payload={"text": "does exercise improve sleep?"},
                )
                assert applied.status == "applied"
                assert applied.state.has("question")

                # 3. Free navigation through enabled transitions (stubs).
                for artifact_id in (
                    "raw_data",
                    "latent_structure",
                    "measurement_structure",
                    "measurements",
                ):
                    outcome = await propose(RunArtifact(artifact_id=artifact_id))
                    assert outcome.status == "applied", (artifact_id, outcome)

                # 4. Typed stage failure: raised, state unchanged.
                before = (await handle.query(EpisodeWorkflow.get_state)).current
                raised = await propose(RunArtifact(artifact_id="statistical_model_spec"))
                assert raised.status == "raised"
                assert raised.error_type == "ModelCompileError"
                assert "report" in raised.diagnostics
                assert raised.diagnostics["checkpoint_ref"].startswith("model-spec-checkpoint:")
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
                assert "measurement_structure" in stale
                assert "raw_data" not in stale  # not derived from question

                # 6. Journal recorded every attempt, including the rejection
                #    and the typed failure.
                records = EpisodeJournal(workspace_id).read_all()
                statuses = [record.status for record in records]
                assert statuses == [
                    "rejected",
                    "applied",  # question
                    "applied",  # ingestion
                    "applied",  # latent-structure
                    "applied",  # measurement-structure
                    "applied",  # extraction
                    "raised",  # model-spec
                    "applied",  # question rewrite
                ]
                assert records[-2].error_type == "ModelCompileError"
                assert "report" in records[-2].diagnostics
                assert (
                    records[-2].diagnostics["checkpoint_ref"].startswith("model-spec-checkpoint:")
                )

                # Store kept both question versions (append-only).
                assert ArtifactStore(workspace_id).list_versions("question") == [1, 2]
        finally:
            await env.shutdown()

    asyncio.run(scenario())
