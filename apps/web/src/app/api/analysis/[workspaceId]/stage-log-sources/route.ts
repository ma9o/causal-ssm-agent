import { NextResponse } from "next/server";
import { type StageId, STAGES } from "@causal-ssm/api-types";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { fetchStageLogFlowRunIds } from "../../_shared";

function isStageId(value: string): value is StageId {
  return STAGES.some((stage) => stage.id === value);
}

export async function GET(request: Request, { params }: { params: Promise<{ workspaceId: string }> }) {
  const { workspaceId } = await params;
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }

  const url = new URL(request.url);
  const stageId = url.searchParams.get("stageId");
  const stageSubflowRunId = url.searchParams.get("stageSubflowRunId");

  if (!stageId || !isStageId(stageId)) {
    return NextResponse.json({ error: "Invalid stageId" }, { status: 400 });
  }

  if (!stageSubflowRunId) {
    return NextResponse.json({ logFlowRunIds: [] });
  }

  const normalizedStageSubflowRunId = stageSubflowRunId.trim();
  if (!normalizedStageSubflowRunId || /[\\/]/.test(normalizedStageSubflowRunId)) {
    return NextResponse.json({ error: "Invalid stageSubflowRunId format" }, { status: 400 });
  }

  try {
    const logFlowRunIds = await fetchStageLogFlowRunIds(stageId, normalizedStageSubflowRunId);
    return NextResponse.json({ logFlowRunIds });
  } catch {
    return NextResponse.json({ error: "Failed to resolve stage log sources" }, { status: 502 });
  }
}
