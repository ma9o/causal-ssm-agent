import { INTERACTIVE_STAGES, STAGE_TOOLS } from "@nof1-causal-lab/api-types";
import { NextResponse } from "next/server";
import { getToolServerUrl } from "@/lib/runtime-urls";
import { isRecord } from "@/lib/utils/type-guards";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

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
 * POST /api/refine/dispatch
 *
 * Direct tool execution from a clicked suggestion chip. Bypasses the LLM —
 * the chip's payload IS the action. Result is returned as JSON; the client
 * is responsible for appending a synthetic dynamic-tool UI message so the
 * LLM sees the result on its next turn.
 *
 * Body: { workspaceId, stageId, tool, input }
 */
export async function POST(req: Request) {
  const body = (await req.json()) as unknown;
  if (!isRecord(body)) {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }
  const { workspaceId, stageId, tool, input } = body;

  if (typeof workspaceId !== "string" || !workspaceId.trim()) {
    return NextResponse.json({ error: "Missing workspaceId" }, { status: 400 });
  }
  const safeStageId = typeof stageId === "string" ? stageId.trim() : "";
  if (!safeStageId || /[\\/]/.test(safeStageId)) {
    return NextResponse.json({ error: "Invalid stageId" }, { status: 400 });
  }
  if (!INTERACTIVE_STAGES.includes(safeStageId)) {
    return NextResponse.json({ error: "Stage is not interactive" }, { status: 400 });
  }
  if (typeof tool !== "string" || !tool.trim()) {
    return NextResponse.json({ error: "Missing tool name" }, { status: 400 });
  }
  if (!isRecord(input)) {
    return NextResponse.json({ error: "input must be an object" }, { status: 400 });
  }

  const toolDef = (STAGE_TOOLS[safeStageId] ?? []).find((t) => t.name === tool);
  if (!toolDef) {
    return NextResponse.json(
      { error: `Tool '${tool}' is not registered for stage ${safeStageId}` },
      { status: 400 },
    );
  }

  const workspaceAccess = await requireWorkspaceAccess(req, workspaceId, {
    requireMutable: true,
  });
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }

  const toolResponse = await fetch(`${TOOL_SERVER}/api/tools/${safeStageId}/${tool}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      workspace_id: workspaceAccess.workspaceId,
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
