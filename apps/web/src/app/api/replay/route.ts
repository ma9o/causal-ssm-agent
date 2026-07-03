import { NextResponse } from "next/server";
import {
  EpisodeRunError,
  proposeMove,
  resolveAutoRunExecOptions,
  STAGE_EDIT_ARTIFACTS,
  startAutoRun,
} from "@/lib/server/episode-runs";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { isRecord } from "@/lib/utils/type-guards";

/**
 * POST /api/replay
 *
 * Writes the edited stage result back into the episode machine as a
 * human-provenance artifact version, then starts the auto-run driver,
 * which recomputes the stages whose outputs went stale.
 *
 * Body: { workspaceId: string, stageId: string, stageData: object }
 */
export async function POST(request: Request) {
  const { workspaceId, stageId, stageData } = await request.json();

  if (!workspaceId || !stageId || !stageData) {
    return NextResponse.json(
      { error: "Missing workspaceId, stageId, or stageData" },
      { status: 400 },
    );
  }

  const safeStageId = typeof stageId === "string" ? stageId.trim() : "";
  if (!safeStageId || /[\\/]/.test(safeStageId)) {
    return NextResponse.json({ error: "Invalid stageId format" }, { status: 400 });
  }
  if (!isRecord(stageData)) {
    return NextResponse.json({ error: "stageData must be an object" }, { status: 400 });
  }

  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId, {
    requireMutable: true,
  });
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: safeWorkspaceId } = workspaceAccess;

  const artifactId = STAGE_EDIT_ARTIFACTS[safeStageId];
  if (!artifactId) {
    return NextResponse.json(
      { error: `Stage ${safeStageId} has no writable artifact` },
      { status: 400 },
    );
  }

  try {
    const outcome = await proposeMove(
      safeWorkspaceId,
      { kind: "write", artifact_id: artifactId, provenance: "human" },
      stageData,
    );
    if (outcome.status === "rejected") {
      return NextResponse.json({ error: outcome.reason ?? "Write move rejected" }, { status: 400 });
    }
    if (outcome.status === "raised") {
      return NextResponse.json(
        { error: outcome.error_message ?? "Write move failed" },
        { status: 502 },
      );
    }

    const options = await resolveAutoRunExecOptions();
    await startAutoRun(safeWorkspaceId, options);

    return NextResponse.json({ ok: true, workspaceId: safeWorkspaceId });
  } catch (err) {
    if (err instanceof EpisodeRunError) {
      if (err.status === 409) {
        return NextResponse.json(
          { error: "A run is already active for this workspace." },
          { status: 409 },
        );
      }
      return NextResponse.json({ error: err.message }, { status: err.status });
    }
    return NextResponse.json(
      { error: `Episode API error: ${err instanceof Error ? err.message : String(err)}` },
      { status: 502 },
    );
  }
}
