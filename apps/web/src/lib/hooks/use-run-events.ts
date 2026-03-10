"use client";

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

export type { PipelineProgress, StageRunStatus, StageTiming } from "./pipeline-progress";

const EVENT_LOOKBACK_MS = 60_000;
const EVENT_LOOKAHEAD_MS = 365 * 24 * 60 * 60 * 1000;

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
    resource?: Record<string, string>;
  };
}

export function buildPrefectEventFilterMessage(runId: string, now = new Date()) {
  return {
    type: "filter",
    filter: {
      event: { prefix: ["prefect.task-run."] },
      related: {
        resources_in_roles: [[`prefect.flow-run.${runId}`, "flow-run"]],
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
      const stage = getStageForPrefectRunName(tr.name);
      if (!stage) continue;

      const status = mapPrefectTaskState(tr.state_type);
      if (!status) continue;
      const startTime = tr.start_time ? new Date(tr.start_time).getTime() : undefined;
      const endTime = tr.end_time ? new Date(tr.end_time).getTime() : undefined;

      if (status === "completed") {
        if (startTime) updateStage(stage.id, "running", startTime);
        updateStage(stage.id, "completed", endTime ?? startTime);
        queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", stage.id] });
      } else if (status === "running") {
        updateStage(stage.id, "running", startTime);
      } else if (status === "failed") {
        if (startTime) updateStage(stage.id, "running", startTime);
        updateStage(stage.id, "failed", endTime ?? startTime);
      }
    }
  } catch {
    // Best-effort — WebSocket will still provide live updates
  }
}

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_DELAY_MS = 1000;

export function useRunEvents(runId: string | null) {
  const queryClient = useQueryClient();
  const wsRef = useRef<ReconnectingWebSocket | null>(null);

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
    queryClient.setQueryData(["pipeline", runId, "status"], initialProgress());
    hydrateFromPrefect(runId, updateStage, queryClient);

    if (isMockMode()) {
      const cleanup = simulatePipelineEvents({
        onStageStart: (id) => updateStage(id, "running"),
        onStageComplete: (id) => {
          updateStage(id, "completed");
          queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", id] });
        },
      });
      return cleanup;
    }

    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:4200";
    const wsUrl = `${apiBase.replace(/^http/, "ws")}/api/events/out`;
    const ws = new ReconnectingWebSocket(wsUrl, ["prefect"], {
      maxRetries: MAX_RECONNECT_ATTEMPTS,
      minReconnectionDelay: BASE_DELAY_MS,
      maxReconnectionDelay: BASE_DELAY_MS * 2 ** MAX_RECONNECT_ATTEMPTS,
      reconnectionDelayGrowFactor: 2,
    });
    wsRef.current = ws;

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

        const data = message.event;
        if (!data) return;

        const taskName = data.resource?.["prefect.resource.name"];
        if (!taskName) return;

        const stage = getStageForPrefectRunName(taskName);
        if (!stage) return;

        // Prefer server-side event timestamp over client-side Date.now()
        const eventTime = data.occurred ? new Date(data.occurred).getTime() : undefined;

        if (data.event === "prefect.task-run.Running") {
          updateStage(stage.id, "running", eventTime);
        } else if (data.event === "prefect.task-run.Completed") {
          updateStage(stage.id, "completed", eventTime);
          queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", stage.id] });
        } else if (data.event === "prefect.task-run.Failed") {
          updateStage(stage.id, "failed", eventTime);
        }

        // Close permanently if pipeline is done — no need to reconnect.
        // Check Prefect task statuses directly (not isFailed, which includes
        // outcome-level failures that don't stop the pipeline when overridden).
        const progress = queryClient.getQueryData<PipelineProgress>(["pipeline", runId, "status"]);
        const anyTaskFailed = progress && STAGES.some((s) => progress.stages[s.id] === "failed");
        if (progress?.isComplete || anyTaskFailed) {
          ws.close();
        }
      } catch {
        // Ignore parse errors
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [runId, queryClient, updateStage]);
}
