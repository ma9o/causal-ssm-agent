import { NextResponse } from "next/server";
import {
  EpisodeRunError,
  resolveAutoRunExecOptions,
  startAutoRun,
  startEpisode,
} from "@/lib/server/episode-runs";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

export async function POST(request: Request) {
  const { workspaceId, query } = await request.json();

  if (typeof workspaceId !== "string" || !workspaceId.trim()) {
    return NextResponse.json({ error: "workspaceId is required" }, { status: 400 });
  }
  if (typeof query !== "string" || !query.trim()) {
    return NextResponse.json({ error: "query is required" }, { status: 400 });
  }

  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId, {
    requireMutable: true,
  });
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }

  try {
    const options = await resolveAutoRunExecOptions();
    await startEpisode(workspaceAccess.workspaceId, query.trim());
    await startAutoRun(workspaceAccess.workspaceId, options);

    return NextResponse.json({ workspaceId: workspaceAccess.workspaceId });
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
