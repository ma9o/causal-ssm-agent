"""Shared byte-stream framing for subprocess harnesses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


class AsyncByteReader(Protocol):
    """Minimal subprocess stdout surface used by the framing loop."""

    async def read(self, n: int = -1) -> bytes: ...


async def drain_newline_delimited_stream(
    stream: AsyncByteReader | None,
    handle_line: Callable[[bytes], None],
) -> None:
    """Read arbitrarily long newline-delimited frames and flush the final frame."""
    if stream is None:
        return

    buffer = bytearray()
    while chunk := await stream.read(65536):
        buffer.extend(chunk)
        while (newline := buffer.find(b"\n")) >= 0:
            handle_line(bytes(buffer[:newline]))
            del buffer[: newline + 1]
    if buffer:
        handle_line(bytes(buffer))
