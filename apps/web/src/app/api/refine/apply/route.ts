import { basename } from "node:path";
import { NextResponse } from "next/server";
import { readData } from "@/lib/storage";

/**
 * POST /api/refine/apply
 *
 * Reads the draft produced by the last successful validation tool call
 * during refinement, merges with the original stage data, and triggers
 * a pipeline replay from that stage.
 *
 * Body: { userId, stageId, rootFlowRunId? }
 */
export async function POST(request: Request) {
  const { userId, stageId, rootFlowRunId } = await request.json();

  if (!userId || !stageId) {
    return NextResponse.json(
      { error: "Missing userId or stageId" },
      { status: 400 },
    );
  }

  const safeUserId = basename(userId.trim());
  const safeStageId = basename(stageId.trim());
  if (!safeUserId || safeUserId !== userId.trim() || safeUserId === "." || safeUserId === "..") {
    return NextResponse.json({ error: "Invalid userId format" }, { status: 400 });
  }
  if (
    !safeStageId ||
    safeStageId !== stageId.trim() ||
    safeStageId === "." ||
    safeStageId === ".."
  ) {
    return NextResponse.json({ error: "Invalid stageId format" }, { status: 400 });
  }

  // Read the draft from the last successful tool call
  let draft: Record<string, unknown>;
  try {
    draft = JSON.parse(await readData(`${safeUserId}/run/${safeStageId}-draft.json`));
  } catch {
    return NextResponse.json(
      {
        error:
          "No draft available. The validation tool must succeed at least once during refinement.",
      },
      { status: 404 },
    );
  }

  // Load original stage data for fields the draft doesn't cover
  let originalDomain: Record<string, unknown>;
  try {
    const raw = await readData(`${safeUserId}/run/${safeStageId}.json`);
    const currentData = JSON.parse(raw);
    // Strip internal fields — keep only domain data
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
    // Trigger replay
    const replayRes = await fetch(new URL("/api/replay", request.url), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        userId: safeUserId,
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
