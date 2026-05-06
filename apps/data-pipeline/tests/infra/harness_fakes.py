"""Reusable subprocess fakes for harness tests."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from causal_ssm_agent.utils.openrouter_client import Tool

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeStdout:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)

    async def read(self, n: int = -1) -> bytes:
        if not self._lines:
            return b""
        if n < 0:
            chunk = b"".join(self._lines)
            self._lines.clear()
            return chunk

        chunk = bytearray()
        while self._lines and len(chunk) < n:
            next_line = self._lines[0]
            remaining = n - len(chunk)
            chunk.extend(next_line[:remaining])
            if remaining >= len(next_line):
                self._lines.pop(0)
            else:
                self._lines[0] = next_line[remaining:]
        return bytes(chunk)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class FakeStderr:
    def __init__(self, text: bytes = b""):
        self._text = text

    async def read(self, n: int = -1) -> bytes:
        if n < 0:
            chunk = self._text
            self._text = b""
            return chunk
        chunk = self._text[:n]
        self._text = self._text[n:]
        return chunk


class FakeProcess:
    def __init__(
        self,
        *,
        lines: list[bytes],
        returncode: int = 0,
        stderr: bytes = b"",
    ):
        self.stdout = FakeStdout(lines)
        self.stderr = FakeStderr(stderr)
        self.returncode = returncode
        self.waited = False

    async def wait(self) -> int:
        self.waited = True
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def jsonl(events: list[dict]) -> list[bytes]:
    return [(json.dumps(event) + "\n").encode() for event in events]


def patch_subprocess(monkeypatch, process_factory: Callable[[list[str]], FakeProcess]):
    captured: dict[str, list[Any]] = {"invocations": [], "envs": []}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["invocations"].append(list(args))
        captured["envs"].append(dict(kwargs.get("env") or {}))
        return process_factory(list(args))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    return captured


def make_terminal_tool(
    *,
    name: str,
    description: str,
    param_name: str | None = None,
    param_description: str = "Terminal payload.",
    success_output: str = "VALID",
) -> Tool:
    async def _execute(**_kwargs) -> str:
        return success_output

    properties = {}
    required = []
    if param_name is not None:
        properties[param_name] = {"type": "string", "description": param_description}
        required.append(param_name)

    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        execute=_execute,
        stop_on_success=True,
        success_output=success_output,
    )
