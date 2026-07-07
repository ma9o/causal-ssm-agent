"""MCP gateway: the episode machine's affordances for an external agent.

A stdio MCP server that adapts the HTTP facade (tool server) for agent
harnesses — Claude Code, claude.ai, or anything MCP-speaking. It is a
thin transport: every tool call becomes an HTTP request to
``TOOL_SERVER_URL`` (default ``http://localhost:8100``), so the agent
drives the exact surface the viewer and curl users see. The only local
knowledge is :func:`describe_machine`, which serves the static artifact
graph and execution classes from the installed package.

Register with Claude Code from the repo root::

    claude mcp add nof1 -- uv run --project apps/data-pipeline nof1-mcp

Navigation loop: ``describe_machine`` once, then ``get_episode`` →
propose (``run_stage`` / ``write_artifact`` / ``start_auto_run``) →
``get_timeline`` for what happened, including rejections and typed
stage errors.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nof1-causal-lab")

_READ_TIMEOUT = httpx.Timeout(30.0)
# A synchronous move blocks until the stage completes (Temporal update).
_MOVE_TIMEOUT = httpx.Timeout(timeout=None, connect=30.0)


def _base_url() -> str:
    return os.environ.get("TOOL_SERVER_URL", "http://localhost:8100").rstrip("/")


async def _request(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: httpx.Timeout = _READ_TIMEOUT,
) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=timeout) as client:
        response = await client.request(method, path, json=json, params=params)
    body = response.json()
    if response.is_error:
        return {
            "error": response.status_code,
            **(body if isinstance(body, dict) else {"detail": body}),
        }
    return body


@mcp.tool()
async def describe_machine() -> dict[str, Any]:
    """The static shape of the episode machine: the artifact transitions (what
    each consumes, produces, optionally co-produces, and derives, plus its
    creation class and whether it is directly writable), the roots, and the
    artifact ids.

    Creation classes: "deterministic" transitions need no credentials;
    "batch_llm" transitions run bulk LLM compute on the service's ambient key;
    "judgment" transitions are proposal work an external agent can do itself by
    writing the produced artifact directly (see write_artifact) — those are the
    ones flagged ``writable``.
    """
    from nof1_causal_lab.machine.artifacts import ARTIFACT_IDS
    from nof1_causal_lab.machine.graph import ARTIFACT_GRAPH, ROOTS, topological_stage_order
    from nof1_causal_lab.machine.hierarchy import describe_actions, describe_contexts

    return {
        "artifact_ids": list(ARTIFACT_IDS),
        "topological_stage_order": topological_stage_order(),
        "contexts": describe_contexts(),
        "actions": describe_actions(),
        "roots": [
            {"artifact_id": root.artifact_id, "write_pins": list(root.write_pins)} for root in ROOTS
        ],
        "stages": [
            {
                "stage_id": spec.stage_id,
                "consumes": list(spec.consumes),
                "produces": list(spec.produces),
                "produces_optional": list(spec.produces_optional),
                "derives": list(spec.derives),
                "creation_class": spec.creation_class,
                "writable": spec.writable,
            }
            for spec in ARTIFACT_GRAPH
        ],
    }


@mcp.tool()
async def get_episode(workspace_id: str) -> dict[str, Any]:
    """Episode state: per-artifact existence/staleness/version/provenance,
    the legal moves at this state, and whether an auto-run is active."""
    return await _request("GET", f"/api/episodes/{workspace_id}")


@mcp.tool()
async def get_timeline(workspace_id: str) -> dict[str, Any]:
    """The transition journal: every move attempt (applied / rejected /
    raised) with typed errors and produced artifact versions."""
    return await _request("GET", f"/api/episodes/{workspace_id}/timeline")


@mcp.tool()
async def get_events(workspace_id: str, after: str | None = None) -> dict[str, Any]:
    """Intra-stage telemetry (stage-2 worker fan-out, stage progress).
    Pass the last seen event id as `after` to page forward."""
    params = {"after": after} if after is not None else None
    return await _request("GET", f"/api/episodes/{workspace_id}/events", params=params)


@mcp.tool()
async def read_artifact(
    workspace_id: str, artifact_id: str, version: int | None = None
) -> dict[str, Any]:
    """An artifact's payload: meta (provenance, derived_from pins) plus all
    JSON payload files inline; binary payload files listed by name. Defaults
    to the episode's current version."""
    params = {"version": version} if version is not None else None
    return await _request(
        "GET", f"/api/episodes/{workspace_id}/artifacts/{artifact_id}", params=params
    )


@mcp.tool()
async def start_episode(workspace_id: str, question: str | None = None) -> dict[str, Any]:
    """Ensure the episode workflow exists; optionally write the `question`
    root artifact. Raw data enters by placing files under
    data/{workspace_id}/input/ before running stage-0."""
    return await _request(
        "POST",
        "/api/episodes",
        json={"workspace_id": workspace_id, "question": question},
        timeout=_MOVE_TIMEOUT,
    )


@mcp.tool()
async def run_stage(workspace_id: str, stage_id: str) -> dict[str, Any]:
    """Propose run(stage). Blocks until the stage finishes and returns the
    move outcome (applied / rejected / raised with the typed error). Long
    stages (stage-4, stage-5b — minutes to hours) can exceed client-side
    tool timeouts: prefer start_auto_run + get_episode polling for those."""
    return await _request(
        "POST",
        f"/api/episodes/{workspace_id}/moves",
        json={"move": {"kind": "run", "stage_id": stage_id}},
        timeout=_MOVE_TIMEOUT,
    )


@mcp.tool()
async def write_artifact(
    workspace_id: str,
    artifact_id: str,
    payload: dict[str, Any],
    provenance: str = "llm",
) -> dict[str, Any]:
    """Propose write(artifact): schema-validated against the artifact's
    contract, journaled, and provenance-stamped ("llm" or "human"). A write
    is a new provenance root — you take responsibility for its content;
    downstream artifacts become stale and re-pin on their next run. Writing
    causal_spec deterministically fans out a recomputed positive
    identification_report when the spec has explicitly identified treatments."""
    return await _request(
        "POST",
        f"/api/episodes/{workspace_id}/moves",
        json={
            "move": {"kind": "write", "artifact_id": artifact_id, "provenance": provenance},
            "payload": payload,
        },
        timeout=_MOVE_TIMEOUT,
    )


@mcp.tool()
async def start_auto_run(workspace_id: str) -> dict[str, Any]:
    """Start the default navigation policy in the background: run enabled
    stages in dependency order while outputs are missing or stale, until
    quiescent or a move fails. Poll get_episode / get_timeline to follow."""
    return await _request(
        "POST", f"/api/episodes/{workspace_id}/auto", json={}, timeout=_MOVE_TIMEOUT
    )


@mcp.tool()
async def list_stage_tools(stage_id: str) -> dict[str, Any]:
    """JSON schemas of a stage's validation/query tools (e.g. stage-6
    simulate, stage-4 submit_model_spec) — the same tools the in-service
    LLM loops use."""
    return await _request("GET", f"/api/tools/{stage_id}")


@mcp.tool()
async def invoke_stage_tool(
    stage_id: str, tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Execute a stage tool against the current artifact store versions.
    Numeric tools (simulate, get_model_info) hard-flag stale provenance
    chains in their warnings — do not report numbers past those flags."""
    return await _request(
        "POST",
        f"/api/tools/{stage_id}/{tool_name}",
        json=arguments,
        timeout=_MOVE_TIMEOUT,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
