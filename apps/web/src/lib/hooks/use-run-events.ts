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
  return { stages, timings: {}, stageOutcomes: {}, currentStage: null, isComplete: false, isFailed: false };
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
          timings[stageId] = { ...timings[stageId]!, completedAt: ts };
        }

        return {
          stages,
          timings,
          stageOutcomes: prev.stageOutcomes,
          currentStage: status === "running" ? stageId : prev.currentStage,
          isComplete: completedAll,
          isFailed: anyFailed || prev.isFailed,
        };
      });
    },
    [queryClient, runId],
  );

  useEffect(() => {
    if (!runId) return;

    // Initialize progress
    queryClient.setQueryData(["pipeline", runId, "status"], initialProgress());

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
    const wsUrl = apiBase.replace(/^http/, "ws") + "/api/events/out";
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
