import { NextResponse } from "next/server";
import { EpisodeRunError, startAutoRun } from "@/lib/server/episode-runs";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

/**
 * POST /api/analysis/[workspaceId]/recompute
 *
 * Starts the auto-run driver to refresh stale artifacts (moves made through
 * other surfaces — facade curl, LLM navigator — leave staleness pending).
 * An already-active auto-run means the recompute is underway, so a facade
 * 409 is treated as success.
 */
export async function POST(
  _request: Request,
  { params }: { params: Promise<{ workspaceId: string }> },
) {
  const { workspaceId } = await params;
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  try {
    await startAutoRun(safeWorkspaceId);
    return NextResponse.json({ ok: true, workspaceId: safeWorkspaceId });
  } catch (error) {
    if (error instanceof EpisodeRunError) {
      if (error.status === 409) {
        return NextResponse.json({ ok: true, workspaceId: safeWorkspaceId });
      }
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json({ error: "Failed to start recompute" }, { status: 502 });
  }
}
