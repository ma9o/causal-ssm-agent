import { isStorageNotFoundError, readBinary } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

/**
 * Map stage IDs to their parquet artifact filenames.
 * Order matters — first existing file wins (see run_store.py STAGE*_FILENAMES).
 */
const PARQUET_MAP: Record<string, string[]> = {
  "stage-0": ["stage0-raw-input.parquet"],
  "stage-2": ["stage2-model-data.parquet"],
};

export async function GET(
  request: Request,
  { params }: { params: Promise<{ workspaceId: string; stage: string }> },
) {
  const { workspaceId, stage } = await params;
  const safeStage = stage.trim();
  if (!safeStage || /[\\/]/.test(safeStage)) {
    return new Response("Invalid route parameters", { status: 400 });
  }
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: safeWorkspaceId } = workspaceAccess;

  const filenames = PARQUET_MAP[safeStage];
  if (!filenames) {
    return new Response("No dataframe available for this stage", { status: 404 });
  }

  for (const filename of filenames) {
    try {
      const bytes = await readBinary(`${safeWorkspaceId}/run/${filename}`);
      return new Response(bytes.buffer as ArrayBuffer, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition": `attachment; filename="${filename}"`,
          "Cache-Control": "private, max-age=3600",
        },
      });
    } catch (e) {
      if (!isStorageNotFoundError(e)) {
        throw e;
      }
    }
  }

  return new Response("Parquet file not found", { status: 404 });
}
