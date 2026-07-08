"""Stage runners executed against the versioned artifact store."""

import polars as pl
import pytest

from nof1_causal_lab.machine.errors import ModelCompileError
from nof1_causal_lab.machine.moves import ExecOptions
from nof1_causal_lab.machine.runners import _run_statistical_model_spec
from nof1_causal_lab.machine.store import ArtifactStore
from tests.helpers import run_async as _run


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


def test_run_statistical_model_spec_raises_model_compile_error_when_spec_does_not_compile(
    workspace, monkeypatch
):
    import nof1_causal_lab.flows.transitions.model_spec.flow as stage4_flow

    store = ArtifactStore(workspace)
    store.write_version(
        "question",
        provenance="human",
        derived_from={},
        produced_by=None,
        json_files={"question.json": {"text": "does stress affect sleep?"}},
    )
    store.write_version(
        "causal_design",
        provenance="llm",
        derived_from={"question": 1},
        produced_by="run:measurement_structure",
        json_files={"causal_design.json": {"causal_design": {"latent": {}}}},
    )
    store.write_version(
        "panel",
        provenance="computed",
        derived_from={"causal_design": 1},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": pl.DataFrame({"indicator": ["m"], "value": [1.0]})},
    )
    store.write_version(
        "validation_report",
        provenance="computed",
        derived_from={"panel": 1},
        produced_by="derive:validation_report",
        json_files={"validation_report.json": {"indicators": {}}},
    )

    async def fake_model_spec_agentic_flow(**_kwargs):
        return {
            "statistical_model_spec": {"likelihoods": [], "parameters": []},
            "authored_priors": {},
        }

    monkeypatch.setattr(stage4_flow, "model_spec_agentic_flow", fake_model_spec_agentic_flow)

    pins = {"question": 1, "causal_design": 1, "panel": 1, "validation_report": 1}
    with pytest.raises(ModelCompileError) as excinfo:
        _run(
            _run_statistical_model_spec(
                workspace, store, pins, ExecOptions(enable_literature=False)
            )
        )

    assert excinfo.value.transition_id == "statistical_model_spec"
    assert "report" in excinfo.value.diagnostics
    # No poisoned pseudo-artifact: the failed attempt writes nothing.
    assert store.list_versions("statistical_model_spec") == []
    assert store.list_versions("compiled_ssm") == []
