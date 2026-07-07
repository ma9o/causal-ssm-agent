import { NextResponse } from "next/server";
import { ArtifactNotFoundError } from "@/lib/server/artifacts";
import { loadStageResult } from "@/lib/stage-result-loader";
import { isStorageNotFoundError } from "@/lib/storage";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ workspaceId: string; stage: string }> },
) {
  const { workspaceId, stage } = await params;

  const safeStage = stage.trim();
  if (!safeStage || /[\\/]/.test(safeStage)) {
    return NextResponse.json({ error: "Invalid route parameters" }, { status: 400 });
  }
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  try {
    return NextResponse.json(await loadStageResult(safeStage, safeWorkspaceId));
  } catch (e: unknown) {
    if (e instanceof ArtifactNotFoundError || isStorageNotFoundError(e)) {
      return NextResponse.json({ error: `No data for ${stage}` }, { status: 404 });
    }
    return NextResponse.json(
      { error: `Failed to read ${stage}: ${e instanceof Error ? e.message : String(e)}` },
      { status: 500 },
    );
  }
}
