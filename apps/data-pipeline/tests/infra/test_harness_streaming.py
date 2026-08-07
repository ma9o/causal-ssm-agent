"""Tests for subprocess stream framing shared by all harnesses."""

from __future__ import annotations

import asyncio

import pytest

from nof1_causal_lab.utils.harness.streaming import drain_newline_delimited_stream


class _ChunkedReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = [*chunks, b""]

    async def read(self, n: int = -1) -> bytes:
        del n
        return self._chunks.pop(0)


def test_drain_stream_frames_split_lines_blanks_and_unterminated_tail() -> None:
    received: list[bytes] = []
    stream = _ChunkedReader([b'{"first":', b"1}\n\n", b'{"second":2}\ntail'])

    asyncio.run(drain_newline_delimited_stream(stream, received.append))

    assert received == [b'{"first":1}', b"", b'{"second":2}', b"tail"]


def test_drain_stream_accepts_frames_larger_than_asyncio_readline_limit() -> None:
    received: list[bytes] = []
    long_frame = b"x" * 70_000

    asyncio.run(
        drain_newline_delimited_stream(
            _ChunkedReader([long_frame[:40_000], long_frame[40_000:] + b"\n"]),
            received.append,
        )
    )

    assert received == [long_frame]


def test_drain_stream_propagates_callback_errors() -> None:
    def fail(_raw: bytes) -> None:
        raise RuntimeError("invalid frame")

    with pytest.raises(RuntimeError, match="invalid frame"):
        asyncio.run(drain_newline_delimited_stream(_ChunkedReader([b"bad\n"]), fail))


def test_drain_stream_accepts_missing_stdout() -> None:
    asyncio.run(drain_newline_delimited_stream(None, lambda _raw: None))
