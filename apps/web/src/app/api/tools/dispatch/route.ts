import { INTERACTIVE_CONTEXTS, CONTEXT_TOOLS } from "@nof1-causal-lab/api-types";
import { NextResponse } from "next/server";
import { getToolServerUrl } from "@/lib/runtime-urls";
import { isRecord } from "@/lib/utils/type-guards";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

const TOOL_SERVER = getToolServerUrl();

async function readToolErrorMessage(response: Response): Promise<string> {
  const bodyText = await response.text();
  if (!bodyText.trim()) {
    return `Tool execution failed with HTTP ${response.status}`;
  }
  try {
    const parsed = JSON.parse(bodyText) as unknown;
    if (isRecord(parsed)) {
      if (typeof parsed.error === "string" && parsed.error.trim()) {
        return parsed.error;
      }
      const detail = parsed.detail;
      if (typeof detail === "string" && detail.trim()) {
        return detail;
      }
      if (isRecord(detail) && typeof detail.message === "string" && detail.message.trim()) {
        return detail.message;
      }
    }
  } catch {
    // Fall through to raw text below.
  }
  return bodyText;
}

/**
 * POST /api/tools/dispatch
 *
 * Direct (no-LLM) context-tool execution — e.g. the ranking interactive DAG's
 * `simulate`. Proxies to the tool server, which reads pinned artifact
 * versions and hard-flags stale provenance.
 *
 * Body: { workspaceId, contextId, tool, input }
 */
export async function POST(req: Request) {
  const body = (await req.json()) as unknown;
  if (!isRecord(body)) {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }
  const { workspaceId, contextId, tool, input } = body;

  const safeWorkspaceId =
    typeof workspaceId === "string" ? normalizeWorkspaceId(workspaceId) : null;
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }
  const safeContextId = typeof contextId === "string" ? contextId.trim() : "";
  if (!safeContextId || /[\\/]/.test(safeContextId)) {
    return NextResponse.json({ error: "Invalid contextId" }, { status: 400 });
  }
  if (!INTERACTIVE_CONTEXTS.includes(safeContextId)) {
    return NextResponse.json({ error: "Context has no dispatchable tools" }, { status: 400 });
  }
  if (typeof tool !== "string" || !tool.trim()) {
    return NextResponse.json({ error: "Missing tool name" }, { status: 400 });
  }
  if (!isRecord(input)) {
    return NextResponse.json({ error: "input must be an object" }, { status: 400 });
  }

  const toolDef = (CONTEXT_TOOLS[safeContextId] ?? []).find((t) => t.name === tool);
  if (!toolDef) {
    return NextResponse.json(
      { error: `Tool '${tool}' is not registered for context ${safeContextId}` },
      { status: 400 },
    );
  }

  const toolResponse = await fetch(`${TOOL_SERVER}/api/tools/${safeContextId}/${tool}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      workspace_id: safeWorkspaceId,
      input,
    }),
  });

  if (!toolResponse.ok) {
    const message = await readToolErrorMessage(toolResponse);
    return NextResponse.json({ error: message }, { status: toolResponse.status });
  }

  const data = (await toolResponse.json()) as { result?: unknown };
  return NextResponse.json({ output: data.result ?? null });
}
