import { NextResponse } from "next/server";
import { loadStageResult } from "@/lib/stage-result-loader";
import { readData } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ workspaceId: string; stage: string }> },
) {
  const { workspaceId, stage } = await params;

  const safeStage = stage.trim();
  if (!safeStage || /[\\/]/.test(safeStage)) {
    return NextResponse.json({ error: "Invalid route parameters" }, { status: 400 });
  }
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: safeWorkspaceId } = workspaceAccess;

  try {
    const raw = await readData(`${safeWorkspaceId}/run/${safeStage}.json`);

    try {
      return NextResponse.json(await loadStageResult(safeStage, raw, safeWorkspaceId));
    } catch (error) {
      return NextResponse.json(
        {
          error: `Invalid persisted data for ${stage}: ${
            error instanceof Error ? error.message : String(error)
          }`,
        },
        { status: 500 },
      );
    }
  } catch {
    return NextResponse.json({ error: `No data for ${stage}` }, { status: 404 });
  }
}
