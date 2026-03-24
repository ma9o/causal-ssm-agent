import { createHash } from "node:crypto";
import { NextResponse } from "next/server";
import {
  findCausalInferenceDeploymentId,
  findLatestWorkspaceRootFlowRunId,
  getPrefectApiBaseUrl,
  launchWorkspaceRootFlowRun,
  PrefectRunError,
  prefectFetch,
} from "@/lib/server/prefect-runs";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

const PREFECT_API = getPrefectApiBaseUrl();
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

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isTerminalFlowState(stateType: unknown): boolean {
  return typeof stateType === "string" && TERMINAL_FLOW_STATES.has(stateType);
}

function slugifyForPrefect(value: string, fallback: string): string {
  const slug = value.replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return slug || fallback;
}

function buildReplayRunName(workspaceId: string, stageId: string): string {
  const workspace = slugifyForPrefect(workspaceId, "workspace").slice(0, 24);
  const stage = slugifyForPrefect(stageId, "stage");
  return `replay-${workspace}-${stage}-${Date.now()}`;
}

function buildReplayIdempotencyKey(
  sourceRootFlowRunId: string | null,
  workspaceId: string,
  stageId: string,
  stageData: unknown,
): string {
  const payload = JSON.stringify({
    sourceRootFlowRunId,
    workspaceId,
    stageId,
    stageData,
  });
  const digest = createHash("sha256").update(payload).digest("hex");
  return `replay:${workspaceId}:${stageId}:${digest}`;
}

async function fetchFlowRun(flowRunId: string): Promise<PrefectFlowRun | null> {
  const res = await prefectFetch(`${PREFECT_API}/flow_runs/${flowRunId}`);
  if (!res.ok) return null;
  return res.json();
}

async function cancelFlowRun(flowRunId: string): Promise<void> {
  const res = await prefectFetch(`${PREFECT_API}/flow_runs/${flowRunId}/set_state`, {
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
 * plus a stage_overrides entry for the refined stage. The new run starts
 * at the edited stage so upstream stages restore from persisted artifacts;
 * the pipeline treats the override as that stage's output and re-runs all
 * downstream stages with the new data.
 *
 * Body: { workspaceId: string, stageId: string, stageData: object, rootFlowRunId?: string }
 */
export async function POST(request: Request) {
  const { workspaceId, stageId, stageData, rootFlowRunId } = await request.json();

  if (!workspaceId || !stageId || !stageData) {
    return NextResponse.json({ error: "Missing workspaceId, stageId, or stageData" }, { status: 400 });
  }

  const safeStageId = typeof stageId === "string" ? stageId.trim() : "";
  const safeRootFlowRunId = isNonEmptyString(rootFlowRunId) ? rootFlowRunId.trim() : null;

  if (!safeStageId || /[\\/]/.test(safeStageId)) {
    return NextResponse.json({ error: "Invalid stageId format" }, { status: 400 });
  }
  if (safeRootFlowRunId && /[\\/]/.test(safeRootFlowRunId)) {
    return NextResponse.json({ error: "Invalid rootFlowRunId format" }, { status: 400 });
  }
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: safeWorkspaceId } = workspaceAccess;

  const stageIdx = STAGE_ORDER.indexOf(safeStageId);
  if (stageIdx === -1) {
    return NextResponse.json({ error: `Unknown stageId: ${safeStageId}` }, { status: 400 });
  }

  try {
    const latestRootFlowRunId = safeRootFlowRunId ?? await findLatestWorkspaceRootFlowRunId(safeWorkspaceId);

    // Build parameters: if we have a prior flow run, reuse its params
    let sourceFlowRun: PrefectFlowRun | null = null;
    let originalParams: Record<string, unknown> = {};
    if (latestRootFlowRunId) {
      sourceFlowRun = await fetchFlowRun(latestRootFlowRunId);
      originalParams = sourceFlowRun?.parameters ?? {};
    }

    // Find the causal-inference deployment
    const deploymentId = await findCausalInferenceDeploymentId();
    if (!deploymentId) {
      return NextResponse.json(
        { error: "causal-inference deployment not found" },
        { status: 404 },
      );
    }
    const existingOverrides =
      (originalParams.stage_overrides as Record<string, unknown>) ?? {};
    const {
      end_stage: _endStage,
      openrouter_access_mode: _oldOpenRouterAccessMode,
      openrouter_secret_ref: _oldOpenRouterSecretRef,
      openrouter_api_key: _oldOpenRouterApiKey,
      start_stage: _startStage,
      ...baseParams
    } = originalParams;
    const replayRunName = buildReplayRunName(safeWorkspaceId, safeStageId);
    const replayIdempotencyKey = buildReplayIdempotencyKey(
      latestRootFlowRunId,
      safeWorkspaceId,
      safeStageId,
      stageData,
    );
    const launch = await launchWorkspaceRootFlowRun({
      beforeActiveRunCheck: async () => {
        if (
          sourceFlowRun &&
          latestRootFlowRunId &&
          !isTerminalFlowState(sourceFlowRun.state?.type)
        ) {
          await cancelFlowRun(latestRootFlowRunId);
          await waitForFlowRunToStop(latestRootFlowRunId);
        }
      },
      context: {
        replay_kind: "stage_override",
        edited_stage_id: safeStageId,
        source_root_flow_run_id: latestRootFlowRunId,
      },
      deploymentId,
      idempotencyKey: replayIdempotencyKey,
      labels: {
        replay: true,
        workspace_id: safeWorkspaceId,
        edited_stage: safeStageId,
        source_root_flow_run_id: latestRootFlowRunId ?? "none",
      },
      name: replayRunName,
      parameters: {
        ...baseParams,
        workspace_id: safeWorkspaceId,
        start_stage: safeStageId,
        stage_overrides: {
          ...existingOverrides,
          [safeStageId]: stageData,
        },
      },
      tags: ["replay", "interactive", safeStageId],
      workspaceId: safeWorkspaceId,
    });
    if (launch.status === "busy") {
      return NextResponse.json(
        {
          error: launch.message,
          ...(launch.rootFlowRunId
            ? { rootFlowRunId: launch.rootFlowRunId }
            : {}),
        },
        { status: 409 },
      );
    }

    const downstreamStart = stageIdx + 1;

    return NextResponse.json({
      ok: true,
      resumeFrom: downstreamStart < STAGE_ORDER.length ? STAGE_ORDER[downstreamStart] : null,
      rootFlowRunId: launch.rootFlowRunId,
    });
  } catch (err) {
    return NextResponse.json(
      { error: `Prefect API error: ${err instanceof Error ? err.message : String(err)}` },
      { status: 502 },
    );
  }
}
