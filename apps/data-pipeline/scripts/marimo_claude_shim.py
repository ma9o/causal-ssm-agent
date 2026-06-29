"""OpenAI-compatible HTTP shim that backs marimo's AI panel with `claude -p`.

marimo's AI features speak HTTP to an OpenAI-compatible endpoint. `claude` is a
CLI, so this shim exposes `/v1/chat/completions` (+ `/v1/models`) and shells out
to `claude -p` for each request. Because `claude` authenticates with the logged-in
Claude Code session, generation bills against the subscription rather than an API key.

Run standalone: `uv run python scripts/marimo_claude_shim.py`
Usually started for you by `scripts/notebooks.sh` (the `notebooks` package script).
"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

PORT = int(os.environ.get("MARIMO_CLAUDE_SHIM_PORT", "8011"))
TIMEOUT_S = float(os.environ.get("MARIMO_CLAUDE_SHIM_TIMEOUT", "180"))

# marimo can only APPEND to its hardcoded "you are a sandboxed notebook copilot"
# system prompt (via `[ai] rules`), never replace it. So we reframe here instead:
# marimo's forwarded system prompt is demoted to an output-format contract, and
# claude is told what it actually is — a full Claude Code agent with tools/skills.
# Override the whole preamble with MARIMO_CLAUDE_SYSTEM.
_PREAMBLE = os.environ.get("MARIMO_CLAUDE_SYSTEM") or (
    "You are a full Claude Code agent operating in this repository with your normal "
    "tools and skills. You are NOT a sandboxed notebook copilot. You are free to read "
    "files, run tools, and invoke skills as needed to satisfy the request.\n\n"
    "The text below is forwarded from the marimo editor. Treat it ONLY as the "
    "output-format contract for your final reply (e.g. wrap code in the requested "
    "fences, do not add prose when it asks for code) — ignore any of its language "
    "implying you are limited to generating code or cannot use tools.\n\n"
    "===== marimo editor output contract ====="
)

app = FastAPI()


async def _run_claude(messages: list[dict]) -> str:
    system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    convo = "\n\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages
        if m.get("role") != "system" and m.get("content")
    )
    append_system = f"{_PREAMBLE}\n\n{system}" if system else _PREAMBLE
    proc = await asyncio.create_subprocess_exec(
        "claude",
        "-p",
        convo,
        "--output-format",
        "json",
        # Non-interactive: never block on a permission prompt that can't be answered.
        "--dangerously-skip-permissions",
        "--append-system-prompt",
        append_system,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"claude -p timed out after {TIMEOUT_S}s") from None
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed: {err.decode(errors='replace')}")
    return json.loads(out.decode())["result"]


def _completion(model: str, text: str) -> dict:
    return {
        "id": "chatcmpl-claude",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
    }


def _sse(model: str, text: str):
    chunk = {
        "id": "chatcmpl-claude",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}],
    }
    done = {
        "id": "chatcmpl-claude",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "claude")
    try:
        text = await _run_claude(body.get("messages", []))
    except Exception as exc:  # noqa: BLE001 — surface any failure to marimo as an OpenAI-style error
        return JSONResponse({"error": {"message": str(exc)}}, status_code=502)
    if body.get("stream"):
        return StreamingResponse(_sse(model, text), media_type="text/event-stream")
    return JSONResponse(_completion(model, text))


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "claude", "object": "model"}]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
