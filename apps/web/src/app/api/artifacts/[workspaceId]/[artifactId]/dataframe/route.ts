import { ArtifactNotFoundError, readArtifactBinary } from "@/lib/server/artifacts";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

const PARQUET_MAP: Record<string, { artifact: "raw_data" | "panel"; key: "raw" | "panel" }> = {
  raw_data: { artifact: "raw_data", key: "raw" },
  measurements: { artifact: "panel", key: "panel" },
};

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ workspaceId: string; artifactId: string }> },
) {
  const { workspaceId, artifactId } = await params;
  const safeArtifactId = artifactId.trim();
  if (!safeArtifactId || /[\\/]/.test(safeArtifactId)) {
    return new Response("Invalid route parameters", { status: 400 });
  }
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return new Response("Invalid workspaceId format", { status: 400 });
  }

  const mapping = PARQUET_MAP[safeArtifactId];
  if (!mapping) {
    return new Response("No dataframe available for this artifact", { status: 404 });
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
    if (e instanceof ArtifactNotFoundError) {
      return new Response("Parquet file not found", { status: 404 });
    }
    throw e;
  }
}
