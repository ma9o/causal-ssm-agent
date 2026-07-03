import { NextResponse } from "next/server";
import {
  EpisodeRunError,
  resolveAutoRunExecOptions,
  startAutoRun,
} from "@/lib/server/episode-runs";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

/**
 * POST /api/analysis/[workspaceId]/recompute
 *
 * Starts the auto-run driver to refresh stale artifacts (moves made through
 * other surfaces — facade curl, LLM navigator — leave staleness pending).
 * An already-active auto-run means the recompute is underway, so a facade
 * 409 is treated as success.
 */
export async function POST(
  request: Request,
  { params }: { params: Promise<{ workspaceId: string }> },
) {
  const { workspaceId } = await params;
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId, {
    requireMutable: true,
  });
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: normalizedWorkspaceId } = workspaceAccess;

  try {
    const options = await resolveAutoRunExecOptions();
    await startAutoRun(normalizedWorkspaceId, options);
    return NextResponse.json({ ok: true, workspaceId: normalizedWorkspaceId });
  } catch (error) {
    if (error instanceof EpisodeRunError) {
      if (error.status === 409) {
        return NextResponse.json({ ok: true, workspaceId: normalizedWorkspaceId });
      }
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json({ error: "Failed to start recompute" }, { status: 502 });
  }
}
