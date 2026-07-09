"""Canonical file-backed LLM trace storage.

Artifacts store only ``llm_trace_ref``. The ref scheme identifies the backend
that owns the trace payload. The current backend is the workspace sidecar tree,
which is published with the rest of the workspace data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.llm import LLMTrace

FILE_TRACE_PREFIX = "file:"


@dataclass(frozen=True)
class TraceMetadata:
    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: str


class TraceStore(Protocol):
    def write_trace(
        self,
        trace: LLMTrace,
        metadata: TraceMetadata,
        *,
        target_path: str | None = None,
    ) -> str: ...

    def read_trace(self, workspace_id: str, ref: str) -> LLMTrace: ...


def _data_relative_path(path: str) -> str:
    base = data_module.DATA_URI.rstrip("/")
    normalized = path.rstrip("/")
    prefix = f"{base}/"
    if not normalized.startswith(prefix):
        raise ValueError(f"Trace path {path!r} is not under data root {data_module.DATA_URI!r}")
    return normalized[len(prefix) :]


def file_trace_ref(path: str) -> str:
    return FILE_TRACE_PREFIX + _data_relative_path(path)


def _file_trace_path(ref: str) -> str:
    if not ref.startswith(FILE_TRACE_PREFIX):
        raise ValueError(f"Not a file trace ref: {ref!r}")
    relative = ref[len(FILE_TRACE_PREFIX) :]
    if relative.startswith("/") or ".." in relative.split("/"):
        raise ValueError(f"Invalid file trace ref path: {ref!r}")
    return storage.join(data_module.DATA_URI, relative)


class FileTraceStore:
    def write_trace(
        self,
        trace: LLMTrace,
        metadata: TraceMetadata,
        *,
        target_path: str | None = None,
    ) -> str:
        path = target_path or storage.join(
            data_module.runs_dir(metadata.workspace_id),
            "traces",
            metadata.run_id,
            f"{metadata.subroutine_id}.json",
        )
        storage.write_text(path, trace.model_dump_json())
        return file_trace_ref(path)

    def read_trace(self, workspace_id: str, ref: str) -> LLMTrace:
        path = _file_trace_path(ref)
        relative = _data_relative_path(path)
        if not relative.startswith(f"{workspace_id}/"):
            raise ValueError(f"Trace ref {ref!r} does not belong to workspace {workspace_id!r}")
        return LLMTrace.model_validate(storage.read_json(path))


def configured_trace_store() -> TraceStore:
    return FileTraceStore()


def write_trace(
    trace: LLMTrace,
    metadata: TraceMetadata,
    *,
    target_path: str | None = None,
) -> str:
    return configured_trace_store().write_trace(trace, metadata, target_path=target_path)


def read_trace(workspace_id: str, ref: str) -> LLMTrace:
    if ref.startswith(FILE_TRACE_PREFIX):
        return FileTraceStore().read_trace(workspace_id, ref)
    raise ValueError(f"Unknown trace ref scheme: {ref!r}")
