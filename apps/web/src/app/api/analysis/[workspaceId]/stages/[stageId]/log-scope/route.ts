import { NextResponse } from "next/server";
import { type StageId, STAGES } from "@causal-ssm/api-types";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { resolveStageLogScopeFlowRunIds } from "../../../../_shared";

function isStageId(value: string): value is StageId {
  return STAGES.some((stage) => stage.id === value);
}

export async function GET(
  request: Request,
  {
    params,
  }: {
    params: Promise<{ workspaceId: string; stageId: string }>;
  },
) {
  const { workspaceId, stageId } = await params;
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }

  if (!isStageId(stageId)) {
    return NextResponse.json({ error: "Invalid stageId" }, { status: 400 });
  }

  const url = new URL(request.url);
  const stageSubflowRunId = url.searchParams.get("stageSubflowRunId");

  if (!stageSubflowRunId) {
    return NextResponse.json({ flowRunIds: [] });
  }

  const normalizedStageSubflowRunId = stageSubflowRunId.trim();
  if (!normalizedStageSubflowRunId || /[\\/]/.test(normalizedStageSubflowRunId)) {
    return NextResponse.json({ error: "Invalid stageSubflowRunId format" }, { status: 400 });
  }

  try {
    const flowRunIds = await resolveStageLogScopeFlowRunIds(stageId, normalizedStageSubflowRunId);
    return NextResponse.json({ flowRunIds });
  } catch {
    return NextResponse.json({ error: "Failed to resolve stage log scope" }, { status: 502 });
  }
}
