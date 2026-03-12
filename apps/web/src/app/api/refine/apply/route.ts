import { readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { NextResponse } from "next/server";

const RESULTS_DIR = process.cwd() + "/../data-pipeline/results";

/**
 * POST /api/refine/apply
 *
 * Reads the draft produced by the last successful validation tool call
 * during refinement, merges with the original stage data, and triggers
 * a pipeline replay from that stage.
 *
 * Body: { runId, stageId }
 */
export async function POST(request: Request) {
  const { runId, stageId } = await request.json();

  if (!runId || !stageId) {
    return NextResponse.json(
      { error: "Missing runId or stageId" },
      { status: 400 },
    );
  }

  const safeRunId = basename(runId);
  const safeStageId = basename(stageId);
  const stagePath = resolve(
    join(RESULTS_DIR, safeRunId, `${safeStageId}.json`),
  );
  const draftPath = resolve(
    join(RESULTS_DIR, safeRunId, `${safeStageId}-draft.json`),
  );

  // Read the draft from the last successful tool call
  let draft: Record<string, unknown>;
  try {
    draft = JSON.parse(await readFile(draftPath, "utf-8"));
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
    const raw = await readFile(stagePath, "utf-8");
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
        runId: safeRunId,
        stageId: safeStageId,
        stageData: merged,
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
