import { getWorkspaceRunTag } from "@/lib/root-flow-runs";
import { getPrefectApiUrl } from "@/lib/runtime-urls";
import {
  createByokSecretRef,
  deleteByokSecretRef,
} from "@/lib/server/byok-secret-store";
import {
  noAccessMessage,
  resolveOpenRouterAccess,
  type RunnableOpenRouterAccess,
  type RunnableOpenRouterAccessMode,
} from "@/lib/server/openrouter-access";
import {
  claimWorkspaceRunSlot,
  releaseWorkspaceRunSlot,
} from "@/lib/server/workspace-run-lock";

const PREFECT_API = getPrefectApiUrl();
const ACTIVE_FLOW_STATE_TYPES = [
  "SCHEDULED",
  "PENDING",
  "RUNNING",
  "PAUSED",
  "CANCELLING",
] as const;
const PREFECT_RETRY_BASE_MS = 500;
const PREFECT_RETRY_MAX_ATTEMPTS = 4;

type PrefectDeployment = { id: string };
type PrefectFlowRunFilterResult = { id: string }[];
type BeforeActiveRunCheck = () => Promise<void>;

export type PreparedWorkspaceRunLaunch =
  | {
      status: "busy";
      message: string;
      rootFlowRunId?: string;
    }
  | {
      status: "ready";
      reservationId: string;
      openrouterAccessMode: RunnableOpenRouterAccessMode;
      openrouterSecretRef: string | null;
    };

export type WorkspaceRootFlowRunLaunchRequest = {
  context?: Record<string, unknown>;
  deploymentId: string;
  idempotencyKey?: string;
  labels?: Record<string, unknown>;
  name?: string;
  parameters: Record<string, unknown>;
  tags?: string[];
  workspaceId: string;
  beforeActiveRunCheck?: BeforeActiveRunCheck;
};

export type WorkspaceRootFlowRunLaunchResult =
  | {
      status: "busy";
      message: string;
      rootFlowRunId?: string;
    }
  | {
      status: "created";
      rootFlowRunId: string;
    };

export class PrefectRunError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
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
  const hintedDelay = parseRetryAfterMs(
    response?.headers.get("Retry-After") ?? null,
  );
  if (hintedDelay != null) {
    return hintedDelay;
  }
  return PREFECT_RETRY_BASE_MS * 2 ** (attempt - 1);
}

export async function prefectFetch(
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

export function getPrefectApiBaseUrl(): string {
  return PREFECT_API;
}

export async function findCausalInferenceDeploymentId(): Promise<
  string | null
> {
  const response = await prefectFetch(`${PREFECT_API}/deployments/filter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      deployments: { name: { any_: ["causal-inference"] } },
    }),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new PrefectRunError(
      502,
      "Failed to find causal-inference deployment",
    );
  }

  const deployments = (await response.json()) as PrefectDeployment[];
  return deployments[0]?.id ?? null;
}

export async function findActiveWorkspaceFlowRunId(
  deploymentId: string,
  workspaceId: string,
): Promise<string | null> {
  const response = await prefectFetch(`${PREFECT_API}/flow_runs/filter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      flow_runs: {
        deployment_id: { any_: [deploymentId] },
        state: { type: { any_: [...ACTIVE_FLOW_STATE_TYPES] } },
        tags: { all_: [getWorkspaceRunTag(workspaceId)] },
      },
      sort: "START_TIME_DESC",
      limit: 1,
    }),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new PrefectRunError(502, "Failed to inspect active workspace runs.");
  }

  const flowRuns = (await response.json()) as PrefectFlowRunFilterResult;
  return flowRuns[0]?.id ?? null;
}

export async function findFlowRunIdByIdempotencyKey(
  deploymentId: string,
  idempotencyKey: string,
): Promise<string | null> {
  const response = await prefectFetch(`${PREFECT_API}/flow_runs/filter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      flow_runs: {
        deployment_id: { any_: [deploymentId] },
        idempotency_key: { any_: [idempotencyKey] },
        parent_task_run_id: { is_null_: true },
      },
      sort: "START_TIME_DESC",
      limit: 1,
    }),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new PrefectRunError(
      502,
      "Failed to inspect idempotent workspace runs.",
    );
  }

  const flowRuns = (await response.json()) as PrefectFlowRunFilterResult;
  return flowRuns[0]?.id ?? null;
}

export async function findLatestWorkspaceRootFlowRunId(
  workspaceId: string,
): Promise<string | null> {
  const response = await prefectFetch(`${PREFECT_API}/flow_runs/filter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      flow_runs: {
        tags: { all_: [getWorkspaceRunTag(workspaceId)] },
        parent_task_run_id: { is_null_: true },
      },
      sort: "START_TIME_DESC",
      limit: 1,
    }),
    cache: "no-store",
  });

  if (!response.ok) {
    throw new PrefectRunError(502, "Failed to inspect workspace runs.");
  }

  const flowRuns = (await response.json()) as PrefectFlowRunFilterResult;
  return flowRuns[0]?.id ?? null;
}

export async function requireRunnableOpenRouterAccess(): Promise<RunnableOpenRouterAccess> {
  const access = await resolveOpenRouterAccess();
  if (access.mode !== "none") {
    return access;
  }

  throw new PrefectRunError(402, noAccessMessage(access.reason));
}

export async function createOpenRouterSecretRefForAccess(
  access: RunnableOpenRouterAccess,
): Promise<string | null> {
  if (access.mode === "local") {
    return null;
  }

  try {
    return await createByokSecretRef(access.apiKey);
  } catch {
    throw new PrefectRunError(500, "OpenRouter secret handoff is not configured.");
  }
}

export async function prepareWorkspaceRunLaunch(
  deploymentId: string,
  workspaceId: string,
  options?: {
    beforeActiveRunCheck?: BeforeActiveRunCheck;
  },
): Promise<PreparedWorkspaceRunLaunch> {
  const access = await requireRunnableOpenRouterAccess();
  const runSlot = await claimWorkspaceRunSlot(workspaceId);
  if (runSlot.status === "busy") {
    return {
      status: "busy",
      message: "A run launch is already in progress for this workspace.",
    };
  }

  const reservationId = runSlot.reservationId;
  let openrouterSecretRef: string | null = null;

  try {
    await options?.beforeActiveRunCheck?.();

    const activeFlowRunId = await findActiveWorkspaceFlowRunId(
      deploymentId,
      workspaceId,
    );
    if (activeFlowRunId) {
      await releaseWorkspaceRunSlotSafely(workspaceId, reservationId);
      return {
        status: "busy",
        message: "A run is already active for this workspace.",
        rootFlowRunId: activeFlowRunId,
      };
    }

    openrouterSecretRef = await createOpenRouterSecretRefForAccess(access);
    return {
      status: "ready",
      reservationId,
      openrouterAccessMode: access.mode,
      openrouterSecretRef,
    };
  } catch (error) {
    await cleanupFailedLaunch(workspaceId, reservationId, openrouterSecretRef);
    throw error;
  }
}

function mergeWorkspaceRunTags(
  workspaceId: string,
  tags: readonly string[] = [],
): string[] {
  return [...new Set([...tags, getWorkspaceRunTag(workspaceId)])];
}

export async function launchWorkspaceRootFlowRun({
  context,
  deploymentId,
  idempotencyKey,
  labels,
  name,
  parameters,
  tags,
  workspaceId,
  beforeActiveRunCheck,
}: WorkspaceRootFlowRunLaunchRequest): Promise<WorkspaceRootFlowRunLaunchResult> {
  let reservationId: string | null = null;
  let openrouterSecretRef: string | null = null;
  let flowRunCreated = false;

  try {
    const launch = await prepareWorkspaceRunLaunch(deploymentId, workspaceId, {
      beforeActiveRunCheck,
    });
    if (launch.status === "busy") {
      return launch;
    }

    reservationId = launch.reservationId;
    openrouterSecretRef = launch.openrouterSecretRef;

    const createResponse = await prefectFetch(
      `${PREFECT_API}/deployments/${deploymentId}/create_flow_run`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...(name ? { name } : {}),
          tags: mergeWorkspaceRunTags(workspaceId, tags),
          ...(idempotencyKey ? { idempotency_key: idempotencyKey } : {}),
          ...(context ? { context } : {}),
          ...(labels ? { labels } : {}),
          parameters: {
            ...parameters,
            openrouter_access_mode: launch.openrouterAccessMode,
            ...(openrouterSecretRef ? { openrouter_secret_ref: openrouterSecretRef } : {}),
          },
        }),
        cache: "no-store",
      },
    );

    if (!createResponse.ok) {
      throw new PrefectRunError(502, "Failed to trigger pipeline");
    }

    const flowRun = (await createResponse.json()) as { id: string };
    flowRunCreated = true;

    await releaseWorkspaceRunSlotSafely(workspaceId, reservationId);
    reservationId = null;

    return {
      status: "created",
      rootFlowRunId: flowRun.id,
    };
  } catch (error) {
    if (!flowRunCreated) {
      await cleanupFailedLaunch(workspaceId, reservationId, openrouterSecretRef);
    }
    throw error;
  }
}

export async function releaseWorkspaceRunSlotSafely(
  workspaceId: string,
  reservationId: string | null,
): Promise<void> {
  if (!reservationId) {
    return;
  }

  try {
    await releaseWorkspaceRunSlot(workspaceId, reservationId);
  } catch {
    // Best-effort cleanup; stale reservations expire automatically.
  }
}

export async function cleanupFailedLaunch(
  workspaceId: string,
  reservationId: string | null,
  openrouterSecretRef: string | null,
): Promise<void> {
  await releaseWorkspaceRunSlotSafely(workspaceId, reservationId);
  if (openrouterSecretRef) {
    await deleteByokSecretRef(openrouterSecretRef);
  }
}
