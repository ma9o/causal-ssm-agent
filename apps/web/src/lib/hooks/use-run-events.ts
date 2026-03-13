"use client";

import type { StageId } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect } from "react";
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

interface PrefectTaskRun {
  name: string;
  state_type: string;
  start_time: string | null;
  end_time: string | null;
}

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

export function buildPrefectEventFilterMessage(flowRunId: string, now = new Date()) {
  return {
    type: "filter",
    filter: {
      // Pipeline emits custom events (stage progress + worker progress)
      // on the root flow run resource.
      event: { prefix: [CAUSAL_SSM_EVENT_PREFIX] },
      resource: {
        id: [`prefect.flow-run.${flowRunId}`],
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
  nTicks: number;
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
    nTicks: (p.n_ticks as number) ?? 0,
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

function applyHydratedTaskRun(
  userId: string,
  taskRun: PrefectTaskRun,
  updateStage: (stageId: StageId, status: StageRunStatus, eventTime?: number, outcome?: string) => void,
  queryClient: ReturnType<typeof useQueryClient>,
) {
  const stage = getStageForPrefectRunName(taskRun.name);
  if (!stage) return;

  const status = mapPrefectTaskState(taskRun.state_type);
  if (!status) return;

  const startTime = taskRun.start_time ? new Date(taskRun.start_time).getTime() : undefined;
  const endTime = taskRun.end_time ? new Date(taskRun.end_time).getTime() : undefined;

  if (status === "completed") {
    if (startTime) updateStage(stage.id, "running", startTime);
    updateStage(stage.id, "completed", endTime ?? startTime);
    invalidateStageData(queryClient, userId, stage.id);
    return;
  }

  if (status === "running") {
    updateStage(stage.id, "running", startTime);
    return;
  }

  if (startTime) updateStage(stage.id, "running", startTime);
  updateStage(stage.id, "failed", endTime ?? startTime);
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
  flowRunId: string,
  updateStage: (stageId: StageId, status: StageRunStatus, eventTime?: number, outcome?: string) => void,
  queryClient: ReturnType<typeof useQueryClient>,
) {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:4200";
  const wsUrl = `${apiBase.replace(/^http/, "ws")}/api/events/out`;
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
        ws.send(JSON.stringify(buildPrefectEventFilterMessage(flowRunId)));
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

/**
 * Fetch existing task runs from Prefect to hydrate progress on page load.
 * Critical for session resumption — without this, stages that completed
 * before the page loaded would stay "pending" forever.
 */
async function hydrateFromPrefect(
  userId: string,
  flowRunId: string,
  updateStage: (stageId: StageId, status: StageRunStatus, eventTime?: number, outcome?: string) => void,
  queryClient: ReturnType<typeof useQueryClient>,
) {
  try {
    const res = await fetch("/prefect/task_runs/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        flow_runs: { id: { any_: [flowRunId] } },
        sort: "EXPECTED_START_TIME_ASC",
      }),
    });
    if (!res.ok) return;

    const taskRuns: PrefectTaskRun[] = await res.json();

    for (const tr of taskRuns) {
      applyHydratedTaskRun(userId, tr, updateStage, queryClient);
    }
  } catch {
    // Best-effort — WebSocket will still provide live updates
  }
}

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_DELAY_MS = 1000;

export function useRunEvents(userId: string | null, flowRunId: string | null) {
  const queryClient = useQueryClient();

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

    // Initialize progress, then hydrate from Prefect to catch up on
    // stages that completed before this page loaded (session resumption).
    queryClient.setQueryData(getPipelineStatusQueryKey(userId), initialProgress());

    if (flowRunId) {
      void hydrateFromPrefect(userId, flowRunId, updateStage, queryClient);
    }

    if (isMockMode()) {
      const cleanup = simulatePipelineEvents({
        onStageStart: (id) => updateStage(id, "running"),
        onStageComplete: (id) => {
          updateStage(id, "completed");
          invalidateStageData(queryClient, userId, id);
        },
      });
      return cleanup;
    }

    if (!flowRunId) return;

    const ws = createRunEventSocket(userId, flowRunId, updateStage, queryClient);

    return () => {
      ws.close();
    };
  }, [userId, flowRunId, queryClient, updateStage]);
}
