import { basename } from "node:path";
import { NextResponse } from "next/server";
import { readData } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

/**
 * POST /api/refine/apply
 *
 * Reads the draft produced by the last successful validation tool call
 * during refinement, merges with the original stage data, and triggers
 * a pipeline replay from that stage.
 *
 * Body: { workspaceId, stageId, rootFlowRunId? }
 */
export async function POST(request: Request) {
  const { workspaceId, stageId, rootFlowRunId } = await request.json();

  if (!workspaceId || !stageId) {
    return NextResponse.json(
      { error: "Missing workspaceId or stageId" },
      { status: 400 },
    );
  }

  const safeStageId = basename(stageId.trim());
  if (
    !safeStageId ||
    safeStageId !== stageId.trim() ||
    safeStageId === "." ||
    safeStageId === ".."
  ) {
    return NextResponse.json({ error: "Invalid stageId format" }, { status: 400 });
  }

  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: safeWorkspaceId } = workspaceAccess;

  let draft: Record<string, unknown>;
  try {
    draft = JSON.parse(await readData(`${safeWorkspaceId}/run/${safeStageId}-draft.json`));
  } catch {
    return NextResponse.json(
      {
        error:
          "No draft available. The validation tool must succeed at least once during refinement.",
      },
      { status: 404 },
    );
  }

  let originalDomain: Record<string, unknown>;
  try {
    const raw = await readData(`${safeWorkspaceId}/run/${safeStageId}.json`);
    const currentData = JSON.parse(raw);
    const {
      llm_trace: _trace,
      outcome: _outcome,
      _live: _liveField,
      ...domain
    } = currentData;
    originalDomain = domain;
  } catch {
    return NextResponse.json(
      { error: "Could not read current stage data" },
      { status: 404 },
    );
  }

  const merged = { ...originalDomain, ...draft };

  try {
    const replayRes = await fetch(new URL("/api/replay", request.url), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        workspaceId: safeWorkspaceId,
        stageId: safeStageId,
        stageData: merged,
        ...(typeof rootFlowRunId === "string" ? { rootFlowRunId } : {}),
      }),
    });

    if (!replayRes.ok) {
      const error = await replayRes.text();
      return NextResponse.json(
        { error: `Replay failed: ${error}` },
        { status: 502 },
      );
    }

    const replayResult = await replayRes.json();
    return NextResponse.json({
      ok: true,
      updatedFields: Object.keys(draft),
      ...replayResult,
    });
  } catch (err) {
    return NextResponse.json(
      {
        error: `Apply failed: ${err instanceof Error ? err.message : String(err)}`,
      },
      { status: 500 },
    );
  }
}
