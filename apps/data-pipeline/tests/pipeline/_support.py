"""Shared pipeline test helpers."""


def redirect_storage(monkeypatch, tmp_path, workspace_id: str = "test_workspace") -> None:
    """Point run storage to tmp_path so tests do not touch real data."""
    del workspace_id

    from nof1_causal_lab.flows import pipeline
    from nof1_causal_lab.flows import run_store as run_store_module
    from nof1_causal_lab.utils import data as data_module

    base = str(tmp_path / "data")

    def _mock_runs_dir(c: str) -> str:
        return f"{base}/{c}/run"

    monkeypatch.setattr(run_store_module, "runs_dir", _mock_runs_dir)
    monkeypatch.setattr(data_module, "runs_dir", _mock_runs_dir)
    monkeypatch.setattr(data_module, "DATA_URI", base)
    monkeypatch.setattr(pipeline, "runs_dir", _mock_runs_dir)
    monkeypatch.setattr(pipeline, "DATA_URI", base)


def noop_artifact(**_kwargs) -> None:
    return None
