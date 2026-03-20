"use client";

import {
  getAnalysisManifestQueryKey,
  type AnalysisStageRuns,
  type AnalysisStageTaskRun,
} from "@/lib/api/analysis";
import { dedupeRootFlowRunIds, getLatestRootFlowRunId } from "@/lib/root-flow-runs";
import { getPrefectEventsUrl } from "@/lib/runtime-urls";
import type { StageId } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import ReconnectingWebSocket from "reconnecting-websocket";
import { isMockMode, simulatePipelineEvents } from "../api/mock-provider";
import { getStageForPrefectRunName } from "../constants/stages";
import {
  applyStageUpdate,
  initialProgress,
  mapPrefectTaskState,
  type PipelineProgress,
  type StageRunStatus,
} from "./pipeline-progress";
import type { Stage2Worker } from "./use-stage2-workers";

export type { PipelineProgress, StageRunStatus, StageTiming } from "./pipeline-progress";

const EVENT_LOOKBACK_MS = 60_000;
const EVENT_LOOKAHEAD_MS = 365 * 24 * 60 * 60 * 1000;
const CAUSAL_SSM_EVENT_PREFIX = "causal-ssm.";
const STAGE_PROGRESS_EVENT_PREFIX = "causal-ssm.pipeline-stage.";
const WORKER_EVENT_PREFIX = "causal-ssm.worker.";

interface PrefectEventSocketMessage {
  type?: string;
  event?: {
    event?: string;
    occurred?: string;
    payload?: Record<string, unknown>;
  };
}

function getPipelineStatusQueryKey(userId: string) {
  return ["pipeline", userId, "status"] as const;
}

function getStageQueryKey(userId: string, stageId: StageId) {
  return ["pipeline", userId, "stage", stageId] as const;
}

function getWorkerQueryKey(userId: string) {
  return ["pipeline", userId, "stage2-workers"] as const;
}

export function buildPrefectEventFilterMessage(rootFlowRunId: string, now = new Date()) {
  return {
    type: "filter",
    filter: {
      // Pipeline emits custom events (stage progress + worker progress)
      // on the root flow run resource.
      event: { prefix: [CAUSAL_SSM_EVENT_PREFIX] },
      resource: {
        id: [`prefect.flow-run.${rootFlowRunId}`],
      },
      // Prefect's websocket filter defaults `occurred.until` to "now".
      // Without an explicit future upper bound, the socket only backfills
      // historical events and drops all subsequent live task transitions.
      occurred: {
        since: new Date(now.getTime() - EVENT_LOOKBACK_MS).toISOString(),
        until: new Date(now.getTime() + EVENT_LOOKAHEAD_MS).toISOString(),
      },
    },
  };
}

function isStageId(value: unknown): value is StageId {
  return typeof value === "string" && STAGES.some((stage) => stage.id === value);
}

function isStageRunStatus(value: unknown): value is StageRunStatus {
  return value === "running" || value === "completed" || value === "failed";
}

export interface StageProgressEvent {
  stageId: StageId;
  status: StageRunStatus;
  eventTime?: number;
  outcome?: string;
  error?: { type: string; message: string };
}

export function parsePrefectStageProgressEvent(
  event: PrefectEventSocketMessage["event"],
): StageProgressEvent | null {
  if (!event?.event?.startsWith(STAGE_PROGRESS_EVENT_PREFIX)) {
    return null;
  }

  const payload = event.payload;
  const stageId = payload?.stage_id;
  const status = payload?.status;
  if (!isStageId(stageId) || !isStageRunStatus(status)) {
    return null;
  }

  return {
    stageId,
    status,
    eventTime: event.occurred ? new Date(event.occurred).getTime() : undefined,
    outcome: typeof payload?.outcome === "string" ? payload.outcome : undefined,
    error:
      payload?.error && typeof payload.error === "object"
        ? (payload.error as { type: string; message: string })
        : undefined,
  };
}

export interface WorkerProgressEvent {
  workerId: number;
  status: "submitted" | "completed" | "failed";
  nWindows: number;
  totalWorkers: number;
  completedCount: number;
  nExtractions?: number;
  nLlmCalls?: number;
  error?: string;
  occurredAt?: number;
}

export function parseWorkerProgressEvent(
  event: PrefectEventSocketMessage["event"],
): WorkerProgressEvent | null {
  if (!event?.event?.startsWith(WORKER_EVENT_PREFIX)) return null;
  const p = event.payload;
  if (!p || typeof p.worker_id !== "number") return null;
  const status = p.status;
  if (status !== "submitted" && status !== "completed" && status !== "failed") return null;
  return {
    workerId: p.worker_id as number,
    status,
    nWindows: (p.n_windows as number) ?? 0,
    totalWorkers: (p.total_workers as number) ?? 0,
    completedCount: (p.completed_count as number) ?? 0,
    nExtractions: typeof p.n_extractions === "number" ? p.n_extractions : undefined,
    nLlmCalls: typeof p.n_llm_calls === "number" ? p.n_llm_calls : undefined,
    error: typeof p.error === "string" ? p.error : undefined,
    occurredAt: event?.occurred ? new Date(event.occurred).getTime() : undefined,
  };
}

function applyWorkerEvent(
  workers: Stage2Worker[],
  event: WorkerProgressEvent,
): Stage2Worker[] {
  const id = `worker-${event.workerId}`;
  const name = `extract-chunk-${event.workerId}`;
  const state: Stage2Worker["state"] =
    event.status === "submitted" ? "running" : event.status;
  const completedAt = (event.status !== "submitted")
    ? (event.occurredAt ?? Date.now())
    : undefined;

  const existing = workers.find((w) => w.id === id);
  if (existing) {
    // Don't regress: if already completed/failed, ignore submitted
    if (event.status === "submitted" && existing.state !== "pending") {
      return workers;
    }
    return workers.map((w) =>
      w.id === id
        ? { ...w, state, nLlmCalls: event.nLlmCalls, completedAt }
        : w,
    );
  }
  return [
    ...workers,
    { id, name, state, nLlmCalls: event.nLlmCalls, completedAt },
  ];
}

function invalidateStageData(
  queryClient: ReturnType<typeof useQueryClient>,
  userId: string,
  stageId: StageId,
) {
  queryClient.invalidateQueries({ queryKey: getStageQueryKey(userId, stageId) });
}

function applyHydratedTaskRunToProgress(
  progress: PipelineProgress,
  taskRun: AnalysisStageTaskRun,
): PipelineProgress {
  const stage = getStageForPrefectRunName(taskRun.name);
  if (!stage) return progress;

  const status = mapPrefectTaskState(taskRun.stateType);
  if (!status) return progress;

  const startTime = taskRun.startTime ? new Date(taskRun.startTime).getTime() : undefined;
  const endTime = taskRun.endTime ? new Date(taskRun.endTime).getTime() : undefined;

  if (status === "completed") {
    let next = progress;
    if (startTime) {
      next = applyStageUpdate(next, stage.id, "running", startTime);
    }
    return applyStageUpdate(next, stage.id, "completed", endTime ?? startTime);
  }

  if (status === "running") {
    return applyStageUpdate(progress, stage.id, "running", startTime);
  }

  let next = progress;
  if (startTime) {
    next = applyStageUpdate(next, stage.id, "running", startTime);
  }
  return applyStageUpdate(next, stage.id, "failed", endTime ?? startTime);
}

function isPipelineTerminal(
  queryClient: ReturnType<typeof useQueryClient>,
  userId: string,
): boolean {
  const progress = queryClient.getQueryData<PipelineProgress>(getPipelineStatusQueryKey(userId));
  return progress?.isComplete === true || progress?.isFailed === true;
}

function createRunEventSocket(
  userId: string,
  rootFlowRunId: string,
  updateStage: (stageId: StageId, status: StageRunStatus, eventTime?: number, outcome?: string) => void,
  queryClient: ReturnType<typeof useQueryClient>,
) {
  // In dev, connect directly to the Prefect server (Next.js rewrites don't proxy WS).
  // In prod, a reverse proxy forwards WS at /prefect/ — derive from window.location.
  const wsUrl = getPrefectEventsUrl(window.location.origin);
  const ws = new ReconnectingWebSocket(wsUrl, ["prefect"], {
    maxRetries: MAX_RECONNECT_ATTEMPTS,
    minReconnectionDelay: BASE_DELAY_MS,
    maxReconnectionDelay: BASE_DELAY_MS * 2 ** MAX_RECONNECT_ATTEMPTS,
    reconnectionDelayGrowFactor: 2,
  });

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "auth", token: null }));
  };

  ws.onmessage = (event: MessageEvent) => {
    try {
      const message = JSON.parse(event.data) as PrefectEventSocketMessage;
      if (message.type === "auth_success") {
        ws.send(JSON.stringify(buildPrefectEventFilterMessage(rootFlowRunId)));
        return;
      }

      // Worker progress events (stage-2 chunks)
      const workerEvent = parseWorkerProgressEvent(message.event);
      if (workerEvent) {
        queryClient.setQueryData<Stage2Worker[]>(
          getWorkerQueryKey(userId),
          (old) => applyWorkerEvent(old ?? [], workerEvent),
        );
        return;
      }

      // Stage progress events
      const stageEvent = parsePrefectStageProgressEvent(message.event);
      if (!stageEvent) return;

      queryClient.invalidateQueries({ queryKey: getAnalysisManifestQueryKey(userId) });
      updateStage(stageEvent.stageId, stageEvent.status, stageEvent.eventTime, stageEvent.outcome);
      if (stageEvent.status === "completed") {
        invalidateStageData(queryClient, userId, stageEvent.stageId);
      }

      if (isPipelineTerminal(queryClient, userId)) {
        ws.close();
      }
    } catch {
      // Ignore parse errors
    }
  };

  return ws;
}

function hydrateFromManifest(
  userId: string,
  stageRuns: AnalysisStageRuns,
  queryClient: ReturnType<typeof useQueryClient>,
) {
  let progress = initialProgress();

  for (const stage of STAGES) {
    const taskRun = stageRuns[stage.id]?.wrapperTaskRun;
    if (!taskRun) {
      continue;
    }

    progress = applyHydratedTaskRunToProgress(progress, taskRun);
    if (progress.stages[stage.id] === "completed") {
      invalidateStageData(queryClient, userId, stage.id);
    }
  }

  return progress;
}

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_DELAY_MS = 1000;

export function useRunEvents(
  userId: string | null,
  rootFlowRunIds: string[],
  stageRuns?: AnalysisStageRuns,
) {
  const queryClient = useQueryClient();
  const activeRootFlowRunId = getLatestRootFlowRunId(rootFlowRunIds);
  const hydratedLineageKeyRef = useRef<string | null>(null);

  const updateStage = useCallback(
    (stageId: StageId, status: StageRunStatus, eventTime?: number, outcome?: string) => {
      queryClient.setQueryData<PipelineProgress>(["pipeline", userId, "status"], (old) =>
        applyStageUpdate(old, stageId, status, eventTime, outcome),
      );
    },
    [queryClient, userId],
  );

  useEffect(() => {
    if (!userId) return;
    const normalizedRootFlowRunIds = dedupeRootFlowRunIds(rootFlowRunIds);
    const lineageKey = `${userId}:${normalizedRootFlowRunIds.join("|")}`;

    if (hydratedLineageKeyRef.current === lineageKey) {
      return;
    }
    hydratedLineageKeyRef.current = lineageKey;

    // Initialize progress, then hydrate from Prefect to catch up on
    // stages that completed before this page loaded (session resumption).
    queryClient.setQueryData(getPipelineStatusQueryKey(userId), initialProgress());

    if (stageRuns) {
      queryClient.setQueryData(
        getPipelineStatusQueryKey(userId),
        hydrateFromManifest(userId, stageRuns, queryClient),
      );
    }
  }, [queryClient, rootFlowRunIds, stageRuns, userId]);

  useEffect(() => {
    if (!userId) return;
    let cancelled = false;

    if (isMockMode()) {
      const cleanup = simulatePipelineEvents({
        onStageStart: (id) => updateStage(id, "running"),
        onStageComplete: (id) => {
          updateStage(id, "completed");
          invalidateStageData(queryClient, userId, id);
        },
      });
      return () => {
        cancelled = true;
        cleanup();
      };
    }

    if (!activeRootFlowRunId) {
      return () => {
        cancelled = true;
      };
    }

    const ws = createRunEventSocket(userId, activeRootFlowRunId, updateStage, queryClient);

    return () => {
      cancelled = true;
      ws.close();
    };
  }, [activeRootFlowRunId, queryClient, updateStage, userId]);
}
