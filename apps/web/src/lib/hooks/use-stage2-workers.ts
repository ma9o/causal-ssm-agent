"use client";

import type { PrefectLogEntry } from "./use-stage-logs";
import type { StageRunStatus } from "./use-run-events";
import { useQuery } from "@tanstack/react-query";

export interface Stage2Worker {
  id: string;
  name: string;
  state: "running" | "completed" | "failed" | "pending";
  nLlmCalls?: number;
  completedAt?: number;
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
  // Find the stage-2 subflow run ID and all its nested flow runs,
  // then fetch logs from all of them (workers run in a nested extraction flow).
  const { fetchStageFlowRunId } = await import("./use-stage-logs");
  const subFlowRunId = await fetchStageFlowRunId(runId, "stage-2");
  if (!subFlowRunId) return [];

  const flowRunIds = [subFlowRunId];

  // Find nested flow runs (stage2-worker-extraction flow is a child)
  try {
    const childRes = await fetch("/prefect/flow_runs/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        flow_runs: { parent_flow_run_id: { any_: [subFlowRunId] } },
        limit: 5,
      }),
    });
    if (childRes.ok) {
      const children: { id: string }[] = await childRes.json();
      flowRunIds.push(...children.map((c) => c.id));
    }
  } catch {
    // Best-effort — still fetch logs from the parent flow
  }

  const res = await fetch("/prefect/logs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      logs: { flow_run_id: { any_: flowRunIds } },
      sort: "TIMESTAMP_ASC",
      limit: 500,
    }),
  });
  if (!res.ok) return [];
  return res.json();
}
