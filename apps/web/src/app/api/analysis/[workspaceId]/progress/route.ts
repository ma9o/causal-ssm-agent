import { NextResponse } from "next/server";
import { getEpisodeEvents, getEpisodeStatus, getEpisodeTimeline } from "@/lib/server/episode-runs";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

/**
 * GET /api/analysis/[workspaceId]/progress?after=<cursor>
 *
 * Server-side proxy over the episode facade for client polling: episode
 * status (auto_running), the transition journal, and intra-stage telemetry
 * events after the given cursor.
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ workspaceId: string }> },
) {
  const { workspaceId } = await params;
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: normalizedWorkspaceId } = workspaceAccess;

  const url = new URL(request.url);
  const after = url.searchParams.get("after");

  try {
    const [status, timeline, events] = await Promise.all([
      getEpisodeStatus(normalizedWorkspaceId),
      getEpisodeTimeline(normalizedWorkspaceId),
      getEpisodeEvents(normalizedWorkspaceId, after),
    ]);

    return NextResponse.json({
      workspaceId: normalizedWorkspaceId,
      autoRunning: status.auto_running,
      seq: status.seq,
      transitions: timeline.transitions,
      events: events.events,
    });
  } catch {
    return NextResponse.json({ error: "Failed to load episode progress" }, { status: 502 });
  }
}
