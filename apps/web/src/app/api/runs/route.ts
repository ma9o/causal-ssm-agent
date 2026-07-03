import { NextResponse } from "next/server";
import { EpisodeRunError, startAutoRun, startEpisode } from "@/lib/server/episode-runs";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

export async function POST(request: Request) {
  const { workspaceId, query } = await request.json();

  if (typeof workspaceId !== "string" || !workspaceId.trim()) {
    return NextResponse.json({ error: "workspaceId is required" }, { status: 400 });
  }
  if (typeof query !== "string" || !query.trim()) {
    return NextResponse.json({ error: "query is required" }, { status: 400 });
  }
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  try {
    await startEpisode(safeWorkspaceId, query.trim());
    await startAutoRun(safeWorkspaceId);

    return NextResponse.json({ workspaceId: safeWorkspaceId });
  } catch (error) {
    if (error instanceof EpisodeRunError) {
      if (error.status === 409) {
        return NextResponse.json(
          { error: "A run is already active for this workspace." },
          { status: 409 },
        );
      }
      return NextResponse.json({ error: error.message }, { status: error.status });
    }

    return NextResponse.json({ error: "Failed to trigger pipeline" }, { status: 502 });
  }
}
