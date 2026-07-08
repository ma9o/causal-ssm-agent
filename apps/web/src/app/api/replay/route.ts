import { NextResponse } from "next/server";
import {
  EpisodeRunError,
  proposeMove,
  startAutoRun,
  WRITABLE_ARTIFACTS,
} from "@/lib/server/episode-runs";
import { isRecord } from "@/lib/utils/type-guards";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

/**
 * POST /api/replay
 *
 * Writes the edited artifact result back into the episode machine as a
 * human-provenance artifact version, then starts the auto-run driver,
 * which recomputes stale downstream artifacts.
 *
 * Body: { workspaceId: string, artifactId: string, payload: object }
 */
export async function POST(request: Request) {
  const { workspaceId, artifactId, payload } = await request.json();

  if (!workspaceId || !artifactId || !payload) {
    return NextResponse.json(
      { error: "Missing workspaceId, artifactId, or payload" },
      { status: 400 },
    );
  }

  const safeArtifactId = typeof artifactId === "string" ? artifactId.trim() : "";
  if (!safeArtifactId || /[\\/]/.test(safeArtifactId)) {
    return NextResponse.json({ error: "Invalid artifactId format" }, { status: 400 });
  }
  if (!isRecord(payload)) {
    return NextResponse.json({ error: "payload must be an object" }, { status: 400 });
  }

  const safeWorkspaceId =
    typeof workspaceId === "string" ? normalizeWorkspaceId(workspaceId) : null;
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  const writableArtifactId = WRITABLE_ARTIFACTS[safeArtifactId];
  if (!writableArtifactId) {
    return NextResponse.json(
      { error: `Artifact ${safeArtifactId} is not writable` },
      { status: 400 },
    );
  }

  try {
    const outcome = await proposeMove(
      safeWorkspaceId,
      { kind: "write", artifact_id: writableArtifactId, provenance: "human" },
      payload,
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

    await startAutoRun(safeWorkspaceId);

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
