import { NextResponse } from "next/server";
import { loadStageResult } from "@/lib/stage-result-loader";
import { isStorageNotFoundError, readData } from "@/lib/storage";
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
  } catch (e: unknown) {
    if (isStorageNotFoundError(e)) {
      return NextResponse.json({ error: `No data for ${stage}` }, { status: 404 });
    }
    return NextResponse.json(
      { error: `Failed to read ${stage}: ${e instanceof Error ? e.message : String(e)}` },
      { status: 500 },
    );
  }
}
