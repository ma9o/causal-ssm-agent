"""Stage runner routing tests."""

import pytest

from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.moves import ExecOptions
from nof1_causal_lab.machine.runners import execute_transition_locally
from tests.helpers import run_async


@pytest.mark.parametrize("artifact_id", ["raw_data", "statistical_model_spec"])
def test_temporal_only_transitions_reject_local_execution(monkeypatch, tmp_path, artifact_id):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "DATA_URI", str(tmp_path / "data"))
    with pytest.raises(RuntimeError, match="Temporal child workflow"):
        run_async(
            execute_transition_locally(
                "test_workspace",
                artifact_id,
                {},
                EpisodeState(),
                ExecOptions(),
            )
        )
