"""Storage helpers for Temporal LLM subroutines."""

from __future__ import annotations

import json
import pickle
from typing import Any

from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage


def subroutine_root(workspace_id: str, run_id: str, subroutine_id: str) -> str:
    return storage.join(data_module.runs_dir(workspace_id), "temporal-llm", run_id, subroutine_id)


def subroutine_conversation_path(
    workspace_id: str,
    run_id: str,
    subroutine_id: str,
    name: str,
) -> str:
    return storage.join(subroutine_root(workspace_id, run_id, subroutine_id), "conversation", name)


def write_subroutine_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


def read_subroutine_json(path: str) -> Any:
    return storage.read_json(path)


def write_subroutine_pickle(path: str, value: Any) -> None:
    with storage.open_file(path, "wb") as file:
        pickle.dump(value, file)


def read_subroutine_pickle(path: str) -> Any:
    with storage.open_file(path, "rb") as file:
        return pickle.load(file)
