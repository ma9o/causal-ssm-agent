import { basename } from "node:path";
import { NextResponse } from "next/server";
import { type StageId, STAGES } from "@causal-ssm/api-types";
import { fetchStageLogFlowRunIds } from "../../_shared";

function isStageId(value: string): value is StageId {
  return STAGES.some((stage) => stage.id === value);
}

export async function GET(request: Request, { params }: { params: Promise<{ userId: string }> }) {
  const { userId } = await params;
  const normalizedUserId = basename(userId);

  if (
    !normalizedUserId ||
    normalizedUserId !== userId ||
    normalizedUserId === "." ||
    normalizedUserId === ".."
  ) {
    return NextResponse.json({ error: "Invalid userId format" }, { status: 400 });
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

  const normalizedStageSubflowRunId = basename(stageSubflowRunId);
  if (
    !normalizedStageSubflowRunId ||
    normalizedStageSubflowRunId !== stageSubflowRunId ||
    normalizedStageSubflowRunId === "." ||
    normalizedStageSubflowRunId === ".."
  ) {
    return NextResponse.json({ error: "Invalid stageSubflowRunId format" }, { status: 400 });
  }

  try {
    const logFlowRunIds = await fetchStageLogFlowRunIds(stageId, normalizedStageSubflowRunId);
    return NextResponse.json({ logFlowRunIds });
  } catch {
    return NextResponse.json({ error: "Failed to resolve stage log sources" }, { status: 502 });
  }
}
