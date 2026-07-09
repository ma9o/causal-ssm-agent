import { NextResponse } from "next/server";
import { ArtifactNotFoundError } from "@/lib/server/artifacts";
import { loadArtifactView } from "@/lib/artifact-view-loader";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ workspaceId: string; artifactId: string }> },
) {
  const { workspaceId, artifactId } = await params;

  const safeArtifactId = artifactId.trim();
  if (!safeArtifactId || /[\\/]/.test(safeArtifactId)) {
    return NextResponse.json({ error: "Invalid route parameters" }, { status: 400 });
  }
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  try {
    return NextResponse.json(await loadArtifactView(safeArtifactId, safeWorkspaceId));
  } catch (e: unknown) {
    if (e instanceof ArtifactNotFoundError) {
      return NextResponse.json({ error: `No data for ${artifactId}` }, { status: 404 });
    }
    return NextResponse.json(
      { error: `Failed to read ${artifactId}: ${e instanceof Error ? e.message : String(e)}` },
      { status: 500 },
    );
  }
}
