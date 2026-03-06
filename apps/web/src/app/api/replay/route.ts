import { writeFile, unlink } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { NextResponse } from "next/server";

const RESULTS_DIR = resolve(process.cwd(), "..", "data-pipeline", "results");
const PREFECT_API = "http://localhost:4200/api";

const STAGE_ORDER = [
  "stage-0",
  "stage-1a",
  "stage-1b",
  "stage-2",
  "stage-3",
  "stage-4",
  "stage-4b",
  "stage-5",
  "stage-6",
];

/**
 * POST /api/replay
 *
 * Overwrites a stage's JSON with refined data, clears downstream stages,
 * and triggers a resume pipeline flow via Prefect.
 *
 * Body: { runId: string, stageId: string, stageData: object }
 */
export async function POST(request: Request) {
  const { runId, stageId, stageData } = await request.json();

  if (!runId || !stageId || !stageData) {
    return NextResponse.json({ error: "Missing runId, stageId, or stageData" }, { status: 400 });
  }

  const safeRunId = basename(runId);
  const safeStageId = basename(stageId);
  const runDir = resolve(join(RESULTS_DIR, safeRunId));

  // Validate paths stay within RESULTS_DIR
  if (!runDir.startsWith(RESULTS_DIR)) {
    return NextResponse.json({ error: "Invalid runId" }, { status: 400 });
  }

  const stageIdx = STAGE_ORDER.indexOf(safeStageId);
  if (stageIdx === -1) {
    return NextResponse.json({ error: `Unknown stageId: ${safeStageId}` }, { status: 400 });
  }

  // 1. Overwrite the stage JSON
  const stagePath = join(runDir, `${safeStageId}.json`);
  await writeFile(stagePath, JSON.stringify(stageData, null, 2), "utf-8");

  // 2. Clear downstream stage JSONs
  const downstreamStart = stageIdx + 1;
  for (const downstream of STAGE_ORDER.slice(downstreamStart)) {
    const path = join(runDir, `${downstream}.json`);
    try {
      await unlink(path);
    } catch {
      // File may not exist — that's fine
    }
  }

  // 3. Determine which stage to resume from (the one after the modified stage)
  const resumeFrom = STAGE_ORDER[downstreamStart];
  if (!resumeFrom) {
    // Modified the last stage — nothing to re-run
    return NextResponse.json({ ok: true, resumeFrom: null });
  }

  // 4. Find the resume-pipeline deployment and trigger it
  try {
    const deploymentsRes = await fetch(`${PREFECT_API}/deployments/filter`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        deployments: { name: { any_: ["resume-pipeline"] } },
      }),
    });

    if (!deploymentsRes.ok) {
      return NextResponse.json(
        { error: "Failed to find resume-pipeline deployment" },
        { status: 502 },
      );
    }

    const deployments = await deploymentsRes.json();
    if (!deployments.length) {
      return NextResponse.json(
        { error: "resume-pipeline deployment not found" },
        { status: 404 },
      );
    }

    const deploymentId = deployments[0].id;

    const flowRunRes = await fetch(
      `${PREFECT_API}/deployments/${deploymentId}/create_flow_run`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          parameters: {
            original_run_id: safeRunId,
            start_from: resumeFrom,
          },
        }),
      },
    );

    if (!flowRunRes.ok) {
      return NextResponse.json(
        { error: "Failed to trigger resume pipeline" },
        { status: 502 },
      );
    }

    const flowRun = await flowRunRes.json();
    return NextResponse.json({
      ok: true,
      resumeFrom,
      flowRunId: flowRun.id,
    });
  } catch (err) {
    return NextResponse.json(
      { error: `Prefect API error: ${err instanceof Error ? err.message : String(err)}` },
      { status: 502 },
    );
  }
}
