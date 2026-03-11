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

export type { PipelineProgress, StageRunStatus, StageTiming } from "./pipeline-progress";

const EVENT_LOOKBACK_MS = 60_000;
const EVENT_LOOKAHEAD_MS = 365 * 24 * 60 * 60 * 1000;
const STAGE_PROGRESS_EVENT_PREFIX = "causal-ssm.pipeline-stage.";

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

function getPipelineStatusQueryKey(runId: string) {
  return ["pipeline", runId, "status"] as const;
}

function getStageQueryKey(runId: string, stageId: StageId) {
  return ["pipeline", runId, "stage", stageId] as const;
}

export function buildPrefectEventFilterMessage(runId: string, now = new Date()) {
  return {
    type: "filter",
    filter: {
      // Prefect task-run events are not scoped to the parent flow-run resource,
      // so the pipeline emits explicit stage progress events on the root flow run.
      event: { prefix: [STAGE_PROGRESS_EVENT_PREFIX] },
      resource: {
        id: [`prefect.flow-run.${runId}`],
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

export function parsePrefectStageProgressEvent(
  event: PrefectEventSocketMessage["event"],
): { stageId: StageId; status: StageRunStatus; eventTime?: number } | null {
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
  };
}

function invalidateStageData(
  queryClient: ReturnType<typeof useQueryClient>,
  runId: string,
  stageId: StageId,
) {
  queryClient.invalidateQueries({ queryKey: getStageQueryKey(runId, stageId) });
}

function applyHydratedTaskRun(
  runId: string,
  taskRun: PrefectTaskRun,
  updateStage: (stageId: StageId, status: StageRunStatus, eventTime?: number) => void,
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
    invalidateStageData(queryClient, runId, stage.id);
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
  runId: string,
): boolean {
  const progress = queryClient.getQueryData<PipelineProgress>(getPipelineStatusQueryKey(runId));
  return progress?.isComplete === true || progress?.isFailed === true;
}

function createRunEventSocket(
  runId: string,
  updateStage: (stageId: StageId, status: StageRunStatus, eventTime?: number) => void,
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
        ws.send(JSON.stringify(buildPrefectEventFilterMessage(runId)));
        return;
      }

      const stageEvent = parsePrefectStageProgressEvent(message.event);
      if (!stageEvent) return;

      updateStage(stageEvent.stageId, stageEvent.status, stageEvent.eventTime);
      if (stageEvent.status === "completed") {
        invalidateStageData(queryClient, runId, stageEvent.stageId);
      }

      if (isPipelineTerminal(queryClient, runId)) {
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
  runId: string,
  updateStage: (stageId: StageId, status: StageRunStatus, eventTime?: number) => void,
  queryClient: ReturnType<typeof useQueryClient>,
) {
  try {
    const res = await fetch("/prefect/task_runs/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        flow_runs: { id: { any_: [runId] } },
        sort: "EXPECTED_START_TIME_ASC",
      }),
    });
    if (!res.ok) return;

    const taskRuns: PrefectTaskRun[] = await res.json();

    for (const tr of taskRuns) {
      applyHydratedTaskRun(runId, tr, updateStage, queryClient);
    }
  } catch {
    // Best-effort — WebSocket will still provide live updates
  }
}

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_DELAY_MS = 1000;

export function useRunEvents(runId: string | null) {
  const queryClient = useQueryClient();

  const updateStage = useCallback(
    (stageId: StageId, status: StageRunStatus, eventTime?: number) => {
      queryClient.setQueryData<PipelineProgress>(["pipeline", runId, "status"], (old) =>
        applyStageUpdate(old, stageId, status, eventTime),
      );
    },
    [queryClient, runId],
  );

  useEffect(() => {
    if (!runId) return;

    // Initialize progress, then hydrate from Prefect to catch up on
    // stages that completed before this page loaded (session resumption).
    queryClient.setQueryData(getPipelineStatusQueryKey(runId), initialProgress());
    void hydrateFromPrefect(runId, updateStage, queryClient);

    if (isMockMode()) {
      const cleanup = simulatePipelineEvents({
        onStageStart: (id) => updateStage(id, "running"),
        onStageComplete: (id) => {
          updateStage(id, "completed");
          invalidateStageData(queryClient, runId, id);
        },
      });
      return cleanup;
    }

    const ws = createRunEventSocket(runId, updateStage, queryClient);

    return () => {
      ws.close();
    };
  }, [runId, queryClient, updateStage]);
}
