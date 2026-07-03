import { NextResponse } from "next/server";
import { getFacadeCapabilities } from "@/lib/server/episode-runs";
import { normalizeWorkspaceId } from "@/lib/workspace-id";
import { buildAnalysisManifest } from "../_shared";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ workspaceId: string }> },
) {
  const { workspaceId } = await params;
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  try {
    const [manifest, capabilities] = await Promise.all([
      buildAnalysisManifest(safeWorkspaceId),
      getFacadeCapabilities(),
    ]);
    if (manifest) {
      return NextResponse.json({ ...manifest, readOnly: !capabilities.moves_enabled });
    }
  } catch {
    // Fall through
  }

  return NextResponse.json({ error: "Analysis manifest not found" }, { status: 404 });
}
