import { NextResponse } from "next/server";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { buildStage4ReplayState } from "../../_shared";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ workspaceId: string }> },
) {
  const { workspaceId } = await params;
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }

  const url = new URL(request.url);
  const rootFlowRunId = url.searchParams.get("rootFlowRunId");
  if (!rootFlowRunId) {
    return NextResponse.json({ graph: null, snapshot: null });
  }

  try {
    return NextResponse.json(await buildStage4ReplayState(rootFlowRunId));
  } catch {
    return NextResponse.json({ graph: null, snapshot: null });
  }
}
