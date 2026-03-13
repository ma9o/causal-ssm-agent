import { basename } from "node:path";
import { NextResponse } from "next/server";
import { readSessions, writeSessions } from "../sessions/_shared";

const PREFECT_API = "http://localhost:4200/api";
const TERMINAL_FLOW_STATES = new Set(["COMPLETED", "FAILED", "CANCELLED", "CRASHED"]);
const CANCELLATION_POLL_MS = 1000;
const CANCELLATION_TIMEOUT_MS = 60000;

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

interface PrefectFlowRun {
  id: string;
  parameters?: Record<string, unknown>;
  state?: {
    type?: string;
    name?: string;
  } | null;
}

interface PrefectSetStateResponse {
  status?: string;
  details?: {
    reason?: string;
  } | null;
}

function isTerminalFlowState(stateType: unknown): boolean {
  return typeof stateType === "string" && TERMINAL_FLOW_STATES.has(stateType);
}

async function fetchFlowRun(flowRunId: string): Promise<PrefectFlowRun | null> {
  const res = await fetch(`${PREFECT_API}/flow_runs/${flowRunId}`);
  if (!res.ok) return null;
  return res.json();
}

async function cancelFlowRun(flowRunId: string): Promise<void> {
  const res = await fetch(`${PREFECT_API}/flow_runs/${flowRunId}/set_state`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      state: { type: "CANCELLING", name: "Cancelling" },
      force: false,
    }),
  });

  if (!res.ok) {
    throw new Error(`Could not cancel flow run ${flowRunId}: ${res.status}`);
  }

  const result: PrefectSetStateResponse = await res.json();
  if (result.status === "ABORT" || result.status === "REJECT") {
    throw new Error(result.details?.reason ?? `Prefect rejected cancellation for ${flowRunId}`);
  }
}

async function waitForFlowRunToStop(flowRunId: string): Promise<PrefectFlowRun> {
  const deadline = Date.now() + CANCELLATION_TIMEOUT_MS;

  while (true) {
    const flowRun = await fetchFlowRun(flowRunId);
    if (!flowRun) {
      throw new Error(`Could not reload flow run ${flowRunId} while waiting for cancellation`);
    }

    if (isTerminalFlowState(flowRun.state?.type)) {
      return flowRun;
    }

    if (Date.now() >= deadline) {
      const stateLabel = flowRun.state?.type ?? flowRun.state?.name ?? "unknown";
      throw new Error(`Timed out waiting for flow run ${flowRunId} to stop (state: ${stateLabel})`);
    }

    await new Promise((resolve) => setTimeout(resolve, CANCELLATION_POLL_MS));
  }
}

/**
 * POST /api/replay
 *
 * Triggers a new pipeline run with the same parameters as the original,
 * plus a stage_overrides entry for the refined stage. The pipeline
 * treats that override as the stage output, skips the overridden stage's
 * computation, and re-runs all downstream stages with the new data.
 *
 * Body: { userId: string, stageId: string, stageData: object }
 */
export async function POST(request: Request) {
  const { userId, stageId, stageData } = await request.json();

  if (!userId || !stageId || !stageData) {
    return NextResponse.json({ error: "Missing userId, stageId, or stageData" }, { status: 400 });
  }

  const safeUserId = basename(userId.trim());
  const safeStageId = basename(stageId);

  if (!safeUserId || safeUserId !== userId.trim() || safeUserId === "." || safeUserId === "..") {
    return NextResponse.json({ error: "Invalid userId format" }, { status: 400 });
  }

  const stageIdx = STAGE_ORDER.indexOf(safeStageId);
  if (stageIdx === -1) {
    return NextResponse.json({ error: `Unknown stageId: ${safeStageId}` }, { status: 400 });
  }

  try {
    // Look up the session to find the flowRunId for fetching original parameters
    const sessions = await readSessions();
    const session = sessions[safeUserId];
    const flowRunId = session?.flowRunId;

    // Build parameters: if we have a prior flow run, reuse its params
    let originalParams: Record<string, unknown> = {};
    if (flowRunId) {
      const flowRun = await fetchFlowRun(flowRunId);
      if (flowRun) {
        originalParams = flowRun.parameters ?? {};
        if (!isTerminalFlowState(flowRun.state?.type)) {
          await cancelFlowRun(flowRunId);
          await waitForFlowRunToStop(flowRunId);
        }
      }
    }

    // Ensure userId is set and add stage_overrides
    const existingOverrides =
      (originalParams.stage_overrides as Record<string, unknown>) ?? {};
    const newParams = {
      ...originalParams,
      user_id: safeUserId,
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
    sessions[safeUserId] = {
      createdAt: session?.createdAt ?? new Date().toISOString(),
      flowRunId: newFlowRun.id,
    };
    await writeSessions(sessions);
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
