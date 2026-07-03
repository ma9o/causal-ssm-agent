import { NextResponse } from "next/server";
import { getEpisodeEvents, getEpisodeStatus, getEpisodeTimeline } from "@/lib/server/episode-runs";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

/**
 * GET /api/analysis/[workspaceId]/progress?after=<cursor>
 *
 * Server-side proxy over the episode facade for client polling: episode
 * status (auto_running plus the per-artifact freshness report), the
 * transition journal, and intra-stage telemetry events after the given
 * cursor.
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ workspaceId: string }> },
) {
  const { workspaceId } = await params;
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  const url = new URL(request.url);
  const after = url.searchParams.get("after");

  try {
    const [status, timeline, events] = await Promise.all([
      getEpisodeStatus(safeWorkspaceId),
      getEpisodeTimeline(safeWorkspaceId),
      getEpisodeEvents(safeWorkspaceId, after),
    ]);

    return NextResponse.json({
      workspaceId: safeWorkspaceId,
      autoRunning: status.auto_running,
      seq: status.seq,
      artifacts: status.artifacts,
      transitions: timeline.transitions,
      events: events.events,
    });
  } catch {
    return NextResponse.json({ error: "Failed to load episode progress" }, { status: 502 });
  }
}
