"use client";

import type { StageId, StageOutcome, StageStatus } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import ReconnectingWebSocket from "reconnecting-websocket";
import { isMockMode, simulatePipelineEvents } from "../api/mock-provider";

export type StageRunStatus = Exclude<StageStatus, "blocked">;

export interface StageTiming {
  startedAt: number;
  completedAt?: number;
}

export interface PipelineProgress {
  stages: Record<StageId, StageRunStatus>;
  timings: Partial<Record<StageId, StageTiming>>;
  stageOutcomes: Partial<Record<StageId, StageOutcome>>;
  currentStage: StageId | null;
  isComplete: boolean;
  isFailed: boolean;
}

function initialProgress(): PipelineProgress {
  const stages = {} as Record<StageId, StageRunStatus>;
  for (const s of STAGES) stages[s.id] = "pending";
  return {
    stages,
    timings: {},
    stageOutcomes: {},
    currentStage: null,
    isComplete: false,
    isFailed: false,
  };
}

interface PrefectTaskRun {
  name: string;
  state_type: string;
  start_time: string | null;
  end_time: string | null;
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
      const stage = STAGES.find((s) => tr.name.startsWith(s.prefectTaskName));
      if (!stage) continue;

      const stateType = tr.state_type.toUpperCase();
      const startTime = tr.start_time ? new Date(tr.start_time).getTime() : undefined;
      const endTime = tr.end_time ? new Date(tr.end_time).getTime() : undefined;

      if (stateType === "COMPLETED") {
        if (startTime) updateStage(stage.id, "running", startTime);
        updateStage(stage.id, "completed", endTime);
        queryClient.invalidateQueries({ queryKey: ["pipeline", runId, "stage", stage.id] });
      } else if (stateType === "RUNNING") {
        updateStage(stage.id, "running", startTime);
      } else if (stateType === "FAILED") {
        if (startTime) updateStage(stage.id, "running", startTime);
        updateStage(stage.id, "failed", endTime);
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
      queryClient.setQueryData<PipelineProgress>(["pipeline", runId, "status"], (old) => {
        const prev = old ?? initialProgress();
        const stages = { ...prev.stages, [stageId]: status };
        const completedAll = STAGES.every((s) => stages[s.id] === "completed");
        const anyFailed = STAGES.some((s) => stages[s.id] === "failed");

        // Use server event timestamp if available, otherwise fall back to client time
        const ts = eventTime ?? Date.now();
        const timings = { ...prev.timings };
        if (status === "running") {
          timings[stageId] = { startedAt: ts };
        } else if ((status === "completed" || status === "failed") && timings[stageId]) {
          const existing = timings[stageId];
          timings[stageId] = { ...existing, completedAt: ts };
        }

        // Advance currentStage without regressing: when a stage completes,
        // point to the next stage so the loading indicator stays current
        // even before the next persist task starts.
        const stageIdx = STAGES.findIndex((s) => s.id === stageId);
        const curIdx = prev.currentStage
          ? STAGES.findIndex((s) => s.id === prev.currentStage)
          : -1;
        let currentStage: StageId | null;
        if (status === "running" && stageIdx >= curIdx) {
          currentStage = stageId;
        } else if (
          status === "completed" &&
          !completedAll &&
          stageIdx + 1 < STAGES.length &&
          stageIdx + 1 > curIdx
        ) {
          currentStage = STAGES[stageIdx + 1].id;
        } else {
          currentStage = prev.currentStage;
        }

        return {
          stages,
          timings,
          stageOutcomes: prev.stageOutcomes,
          currentStage,
          isComplete: completedAll,
          isFailed: anyFailed || prev.isFailed,
        };
      });
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
    const ws = new ReconnectingWebSocket(wsUrl, [], {
      maxRetries: MAX_RECONNECT_ATTEMPTS,
      minReconnectionDelay: BASE_DELAY_MS,
      maxReconnectionDelay: BASE_DELAY_MS * 2 ** MAX_RECONNECT_ATTEMPTS,
      reconnectionDelayGrowFactor: 2,
    });
    wsRef.current = ws;

    ws.onopen = () => {
      // Prefect's /api/events/out requires a filter message before it streams events.
      // Without this, the server waits indefinitely and sends nothing.
      ws.send(
        JSON.stringify({
          type: "filter",
          filter: {
            event: { prefix: ["prefect.task-run."] },
            related: {
              resources_in_roles: [[`prefect.flow-run.${runId}`, "flow-run"]],
            },
          },
        }),
      );
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);

        const taskName = data.resource?.["prefect.task-run.name"];
        if (!taskName) return;

        const stage = STAGES.find((s) => taskName.startsWith(s.prefectTaskName));
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
