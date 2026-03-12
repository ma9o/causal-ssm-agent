"use client";

import type { PrefectLogEntry } from "./use-stage-logs";
import type { StageRunStatus } from "./use-run-events";
import { useQuery } from "@tanstack/react-query";

export interface Stage2Worker {
  id: string;
  name: string;
  state: "running" | "completed" | "failed" | "pending";
}

export interface Stage2WorkerProgress {
  workers: Stage2Worker[];
  logs: PrefectLogEntry[];
}

/**
 * Stage-2 worker progress via WebSocket events + log polling.
 *
 * Worker states (submitted/completed/failed) arrive over the existing
 * WebSocket connection in use-run-events.ts and are written into the
 * ["pipeline", runId, "stage2-workers"] query cache key.
 *
 * Logs must still be polled — Prefect has no log WebSocket API.
 */
export function useStage2Workers(
  runId: string,
  stageStatus: StageRunStatus,
): Stage2WorkerProgress {
  const isActive = stageStatus === "running";

  // Workers: populated by WebSocket events in use-run-events.ts
  const { data: workers = [] } = useQuery<Stage2Worker[]>({
    queryKey: ["pipeline", runId, "stage2-workers"],
    queryFn: () => [],
    enabled: isActive,
    staleTime: Infinity,
  });

  // Logs: still polled (Prefect has no log WebSocket)
  const { data: logs = [] } = useQuery({
    queryKey: ["pipeline", runId, "stage2-logs"],
    queryFn: () => fetchStage2Logs(runId),
    enabled: isActive && workers.length > 0,
    refetchInterval: 3000,
    staleTime: 1000,
  });

  return { workers, logs };
}

async function fetchStage2Logs(runId: string): Promise<PrefectLogEntry[]> {
  // Find the stage-2 subflow run ID, then fetch its logs
  const { fetchStageFlowRunId } = await import("./use-stage-logs");
  const subFlowRunId = await fetchStageFlowRunId(runId, "stage-2");
  if (!subFlowRunId) return [];

  const res = await fetch("/prefect/logs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      logs: { flow_run_id: { any_: [subFlowRunId] } },
      sort: "TIMESTAMP_ASC",
      limit: 500,
    }),
  });
  if (!res.ok) return [];
  return res.json();
}
