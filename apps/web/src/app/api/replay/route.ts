import { createHash } from "node:crypto";
import { basename } from "node:path";
import { NextResponse } from "next/server";
import { getPrefectApiUrl } from "@/lib/runtime-urls";
import {
  appendSessionRootFlowRunId,
  getLatestSessionRootFlowRunId,
  readSession,
  writeSession,
} from "../sessions/_shared";

const PREFECT_API = getPrefectApiUrl();
const TERMINAL_FLOW_STATES = new Set(["COMPLETED", "FAILED", "CANCELLED", "CRASHED"]);
const CANCELLATION_POLL_MS = 1000;
const CANCELLATION_TIMEOUT_MS = 60000;
const PREFECT_RETRY_BASE_MS = 500;
const PREFECT_RETRY_MAX_ATTEMPTS = 4;

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

interface PrefectDeployment {
  id: string;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isTerminalFlowState(stateType: unknown): boolean {
  return typeof stateType === "string" && TERMINAL_FLOW_STATES.has(stateType);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryablePrefectStatus(status: number): boolean {
  return status === 429 || status >= 500;
}

function parseRetryAfterMs(value: string | null): number | null {
  if (!value) return null;

  const seconds = Number(value);
  if (Number.isFinite(seconds)) {
    return Math.max(0, seconds * 1000);
  }

  const retryAt = Date.parse(value);
  if (Number.isNaN(retryAt)) {
    return null;
  }

  return Math.max(0, retryAt - Date.now());
}

function getPrefectRetryDelayMs(attempt: number, response?: Response): number {
  const hintedDelay = parseRetryAfterMs(response?.headers.get("Retry-After") ?? null);
  if (hintedDelay != null) {
    return hintedDelay;
  }
  return PREFECT_RETRY_BASE_MS * 2 ** (attempt - 1);
}

async function prefectFetch(
  input: string,
  init?: RequestInit,
  attempt = 1,
): Promise<Response> {
  try {
    const response = await fetch(input, init);
    if (
      attempt < PREFECT_RETRY_MAX_ATTEMPTS &&
      isRetryablePrefectStatus(response.status)
    ) {
      await sleep(getPrefectRetryDelayMs(attempt, response));
      return prefectFetch(input, init, attempt + 1);
    }
    return response;
  } catch (error) {
    if (attempt >= PREFECT_RETRY_MAX_ATTEMPTS) {
      throw error;
    }
    await sleep(getPrefectRetryDelayMs(attempt));
    return prefectFetch(input, init, attempt + 1);
  }
}

function slugifyForPrefect(value: string, fallback: string): string {
  const slug = value.replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return slug || fallback;
}

function buildReplayRunName(userId: string, stageId: string): string {
  const user = slugifyForPrefect(userId, "user").slice(0, 24);
  const stage = slugifyForPrefect(stageId, "stage");
  return `replay-${user}-${stage}-${Date.now()}`;
}

function buildReplayIdempotencyKey(
  sourceRootFlowRunId: string | null,
  userId: string,
  stageId: string,
  stageData: unknown,
): string {
  const payload = JSON.stringify({
    sourceRootFlowRunId,
    userId,
    stageId,
    stageData,
  });
  const digest = createHash("sha256").update(payload).digest("hex");
  return `replay:${userId}:${stageId}:${digest}`;
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
 * Body: { userId: string, stageId: string, stageData: object, rootFlowRunId?: string }
 */
export async function POST(request: Request) {
  const { userId, stageId, stageData, rootFlowRunId } = await request.json();

  if (!userId || !stageId || !stageData) {
    return NextResponse.json({ error: "Missing userId, stageId, or stageData" }, { status: 400 });
  }

  const safeUserId = basename(userId.trim());
  const safeStageId = basename(stageId);
  const safeRootFlowRunId =
    isNonEmptyString(rootFlowRunId) ? basename(rootFlowRunId.trim()) : null;

  if (!safeUserId || safeUserId !== userId.trim() || safeUserId === "." || safeUserId === "..") {
    return NextResponse.json({ error: "Invalid userId format" }, { status: 400 });
  }
  if (
    safeRootFlowRunId &&
    (safeRootFlowRunId !== rootFlowRunId.trim() ||
      safeRootFlowRunId === "." ||
      safeRootFlowRunId === "..")
  ) {
    return NextResponse.json({ error: "Invalid rootFlowRunId format" }, { status: 400 });
  }

  const stageIdx = STAGE_ORDER.indexOf(safeStageId);
  if (stageIdx === -1) {
    return NextResponse.json({ error: `Unknown stageId: ${safeStageId}` }, { status: 400 });
  }

  try {
    // Use the current page's explicit root flow run when available so replay still
    // works even if session registration failed after the source run launched.
    const session = await readSession(safeUserId) ?? undefined;
    const latestRootFlowRunId = safeRootFlowRunId ?? getLatestSessionRootFlowRunId(session);

    // Build parameters: if we have a prior flow run, reuse its params
    let originalParams: Record<string, unknown> = {};
    if (latestRootFlowRunId) {
      const flowRun = await fetchFlowRun(latestRootFlowRunId);
      if (flowRun) {
        originalParams = flowRun.parameters ?? {};
        if (!isTerminalFlowState(flowRun.state?.type)) {
          await cancelFlowRun(latestRootFlowRunId);
          await waitForFlowRunToStop(latestRootFlowRunId);
        }
      }
    }

    // Ensure the replay resumes from the edited stage boundary and does not
    // inherit stale replay-window bounds from the previous run.
    const existingOverrides =
      (originalParams.stage_overrides as Record<string, unknown>) ?? {};
    const { start_stage: _startStage, end_stage: _endStage, ...baseParams } = originalParams;
    const newParams = {
      ...baseParams,
      user_id: safeUserId,
      start_stage: safeStageId,
      stage_overrides: {
        ...existingOverrides,
        [safeStageId]: stageData,
      },
    };
    const replayRunName = buildReplayRunName(safeUserId, safeStageId);
    const replayIdempotencyKey = buildReplayIdempotencyKey(
      latestRootFlowRunId,
      safeUserId,
      safeStageId,
      stageData,
    );

    // Find the causal-inference deployment
    const deploymentsRes = await prefectFetch(`${PREFECT_API}/deployments/filter`, {
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

    const deployments = (await deploymentsRes.json()) as PrefectDeployment[];
    if (!deployments.length) {
      return NextResponse.json(
        { error: "causal-inference deployment not found" },
        { status: 404 },
      );
    }

    const deploymentId = deployments[0].id;

    // Trigger new flow run with original params + stage override
    const createRes = await prefectFetch(
      `${PREFECT_API}/deployments/${deploymentId}/create_flow_run`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: replayRunName,
          tags: ["replay", "interactive", safeStageId],
          idempotency_key: replayIdempotencyKey,
          context: {
            replay_kind: "stage_override",
            edited_stage_id: safeStageId,
            source_root_flow_run_id: latestRootFlowRunId,
          },
          labels: {
            replay: true,
            user_id: safeUserId,
            edited_stage: safeStageId,
            source_root_flow_run_id: latestRootFlowRunId ?? "none",
          },
          parameters: newParams,
        }),
      },
    );

    if (!createRes.ok) {
      return NextResponse.json({ error: "Failed to trigger pipeline" }, { status: 502 });
    }

    const newFlowRun = await createRes.json();
    let sessionPersisted = true;
    try {
      await writeSession(safeUserId, appendSessionRootFlowRunId(session, newFlowRun.id));
    } catch {
      sessionPersisted = false;
    }
    const downstreamStart = stageIdx + 1;

    return NextResponse.json({
      ok: true,
      resumeFrom: downstreamStart < STAGE_ORDER.length ? STAGE_ORDER[downstreamStart] : null,
      rootFlowRunId: newFlowRun.id,
      sessionPersisted,
    });
  } catch (err) {
    return NextResponse.json(
      { error: `Prefect API error: ${err instanceof Error ? err.message : String(err)}` },
      { status: 502 },
    );
  }
}
