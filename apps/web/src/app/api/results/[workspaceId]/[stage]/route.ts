import { NextResponse } from "next/server";
import { isSharedWorkspaceId } from "@/lib/shared-workspaces";
import { loadStageResult } from "@/lib/stage-result-loader";
import { isStorageNotFoundError, readData } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

const SHARED_RESULT_CACHE_CONTROL =
  "public, max-age=120, s-maxage=86400, stale-while-revalidate=604800";

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
      const response = NextResponse.json(await loadStageResult(safeStage, raw, safeWorkspaceId));
      if (isSharedWorkspaceId(safeWorkspaceId)) {
        response.headers.set("Cache-Control", SHARED_RESULT_CACHE_CONTROL);
      }
      return response;
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
