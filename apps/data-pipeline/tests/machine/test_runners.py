"""Stage runners executed against the versioned artifact store."""

import polars as pl
import pytest

from nof1_causal_lab.machine.errors import ModelCompileError
from nof1_causal_lab.machine.moves import ExecOptions
from nof1_causal_lab.machine.runners import _run_stage4
from nof1_causal_lab.machine.store import ArtifactStore
from tests.helpers import run_async as _run


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


def test_run_stage4_raises_model_compile_error_when_spec_does_not_compile(workspace, monkeypatch):
    import nof1_causal_lab.flows.stages.stage4.flow as stage4_flow

    store = ArtifactStore(workspace)
    store.write_version(
        "question",
        provenance="human",
        derived_from={},
        produced_by=None,
        json_files={"question.json": {"text": "does stress affect sleep?"}},
    )
    store.write_version(
        "causal_spec",
        provenance="llm",
        derived_from={"question": 1},
        produced_by="stage-1b",
        json_files={"causal_spec.json": {"causal_spec": {"latent": {}}}},
    )
    store.write_version(
        "model_data",
        provenance="computed",
        derived_from={"causal_spec": 1},
        produced_by="stage-2",
        parquet_files={"model_data.parquet": pl.DataFrame({"indicator": ["m"], "value": [1.0]})},
    )
    store.write_version(
        "validation_report",
        provenance="computed",
        derived_from={"model_data": 1},
        produced_by="stage-3",
        json_files={"validation_report.json": {"indicators": {}}},
    )

    async def fake_stage4_agentic_flow(**_kwargs):
        return {"model_spec": {"likelihoods": [], "parameters": []}, "authored_priors": {}}

    monkeypatch.setattr(stage4_flow, "stage4_agentic_flow", fake_stage4_agentic_flow)

    pins = {"question": 1, "causal_spec": 1, "model_data": 1, "validation_report": 1}
    with pytest.raises(ModelCompileError) as excinfo:
        _run(_run_stage4(workspace, store, pins, ExecOptions(enable_literature=False)))

    assert excinfo.value.stage_id == "stage-4"
    assert "report" in excinfo.value.diagnostics
    # No poisoned pseudo-artifact: the failed attempt writes nothing.
    assert store.list_versions("compiled_ssm") == []
