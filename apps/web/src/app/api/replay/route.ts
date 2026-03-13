import { basename } from "node:path";
import { NextResponse } from "next/server";
import { readSessions } from "../sessions/_shared";

const PREFECT_API = "http://localhost:4200/api";

const STAGE_ORDER = [
  "stage-0",
  "stage-1a",
  "stage-1b",
  "stage-2",
  "stage-3",
  "stage-4",
  "stage-4b",
  "stage-5a",
  "stage-5b",
  "stage-6",
];

/**
 * POST /api/replay
 *
 * Triggers a new pipeline run with the same parameters as the original,
 * plus a stage_overrides entry for the refined stage. The pipeline
 * treats that override as the stage output, skips the overridden stage's
 * computation, and re-runs all downstream stages with the new data.
 *
 * Body: { code: string, stageId: string, stageData: object }
 */
export async function POST(request: Request) {
  const { code, stageId, stageData } = await request.json();

  if (!code || !stageId || !stageData) {
    return NextResponse.json({ error: "Missing code, stageId, or stageData" }, { status: 400 });
  }

  const safeCode = basename(code);
  const safeStageId = basename(stageId);

  const stageIdx = STAGE_ORDER.indexOf(safeStageId);
  if (stageIdx === -1) {
    return NextResponse.json({ error: `Unknown stageId: ${safeStageId}` }, { status: 400 });
  }

  try {
    // Look up the session to find the flowRunId for fetching original parameters
    const sessions = await readSessions();
    const session = sessions[safeCode.toUpperCase()];
    const flowRunId = session?.flowRunId;

    // Build parameters: if we have a prior flow run, reuse its params
    let originalParams: Record<string, unknown> = {};
    if (flowRunId) {
      const flowRunRes = await fetch(`${PREFECT_API}/flow_runs/${flowRunId}`);
      if (flowRunRes.ok) {
        const flowRun = await flowRunRes.json();
        originalParams = flowRun.parameters ?? {};
      }
    }

    // Ensure code is set and add stage_overrides
    const existingOverrides =
      (originalParams.stage_overrides as Record<string, unknown>) ?? {};
    const newParams = {
      ...originalParams,
      code: safeCode,
      stage_overrides: {
        ...existingOverrides,
        [safeStageId]: stageData,
      },
    };

    // Find the causal-inference deployment
    const deploymentsRes = await fetch(`${PREFECT_API}/deployments/filter`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        deployments: { name: { any_: ["causal-inference"] } },
      }),
    });

    if (!deploymentsRes.ok) {
      return NextResponse.json(
        { error: "Failed to find causal-inference deployment" },
        { status: 502 },
      );
    }

    const deployments = await deploymentsRes.json();
    if (!deployments.length) {
      return NextResponse.json(
        { error: "causal-inference deployment not found" },
        { status: 404 },
      );
    }

    const deploymentId = deployments[0].id;

    // Trigger new flow run with original params + stage override
    const createRes = await fetch(
      `${PREFECT_API}/deployments/${deploymentId}/create_flow_run`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ parameters: newParams }),
      },
    );

    if (!createRes.ok) {
      return NextResponse.json({ error: "Failed to trigger pipeline" }, { status: 502 });
    }

    const newFlowRun = await createRes.json();
    const downstreamStart = stageIdx + 1;

    return NextResponse.json({
      ok: true,
      resumeFrom: downstreamStart < STAGE_ORDER.length ? STAGE_ORDER[downstreamStart] : null,
      flowRunId: newFlowRun.id,
    });
  } catch (err) {
    return NextResponse.json(
      { error: `Prefect API error: ${err instanceof Error ? err.message : String(err)}` },
      { status: 502 },
    );
  }
}
