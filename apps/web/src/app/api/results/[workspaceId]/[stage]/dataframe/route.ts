import { ArtifactNotFoundError, readArtifactBinary } from "@/lib/server/artifacts";
import { isStorageNotFoundError } from "@/lib/storage";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

/**
 * Map stage IDs to their canonical parquet artifact.
 */
const PARQUET_MAP: Record<string, { artifact: "raw_data" | "panel"; key: "raw" | "panel" }> = {
  "stage-0": { artifact: "raw_data", key: "raw" },
  "stage-2": { artifact: "panel", key: "panel" },
};

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ workspaceId: string; stage: string }> },
) {
  const { workspaceId, stage } = await params;
  const safeStage = stage.trim();
  if (!safeStage || /[\\/]/.test(safeStage)) {
    return new Response("Invalid route parameters", { status: 400 });
  }
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return new Response("Invalid workspaceId format", { status: 400 });
  }

  const mapping = PARQUET_MAP[safeStage];
  if (!mapping) {
    return new Response("No dataframe available for this stage", { status: 404 });
  }

  try {
    const { data, filename } = await readArtifactBinary(
      safeWorkspaceId,
      mapping.artifact,
      "parquet",
      mapping.key,
    );
    return new Response(data.slice(), {
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": `attachment; filename="${filename}"`,
        "Cache-Control": "private, max-age=3600",
      },
    });
  } catch (e) {
    if (e instanceof ArtifactNotFoundError || isStorageNotFoundError(e)) {
      return new Response("Parquet file not found", { status: 404 });
    }
    throw e;
  }
}
