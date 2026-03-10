"use client";

import { fetchStageFlowRunId, type PrefectLogEntry } from "./use-stage-logs";
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
  subFlowRunId: string | null;
}

interface PrefectTaskRun {
  id: string;
  name: string;
  state_type: string;
}

async function fetchStage2Workers(
  subFlowRunId: string,
): Promise<Stage2Worker[]> {
  const res = await fetch("/prefect/task_runs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      flow_runs: { id: { any_: [subFlowRunId] } },
      sort: "EXPECTED_START_TIME_ASC",
    }),
  });
  if (!res.ok) return [];

  const taskRuns: PrefectTaskRun[] = await res.json();
  return taskRuns
    .filter((taskRun) => taskRun.name.startsWith("extract-chunk-"))
    .map((taskRun) => ({
      id: taskRun.id,
      name: taskRun.name,
      state: mapState(taskRun.state_type),
    }));
}

function mapState(
  stateType: string,
): "running" | "completed" | "failed" | "pending" {
  switch (stateType.toUpperCase()) {
    case "COMPLETED":
      return "completed";
    case "FAILED":
    case "CRASHED":
    case "CANCELLED":
      return "failed";
    case "RUNNING":
      return "running";
    default:
      return "pending";
  }
}

async function fetchStage2Logs(
  subFlowRunId: string,
): Promise<PrefectLogEntry[]> {
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

export function useStage2Workers(
  runId: string,
  stageStatus: StageRunStatus,
): Stage2WorkerProgress {
  const isActive = stageStatus === "running";

  const { data: subFlowRunId } = useQuery({
    queryKey: ["pipeline", runId, "stage2-subflow"],
    queryFn: () => fetchStageFlowRunId(runId, "stage-2"),
    enabled: isActive,
    refetchInterval: (query) => (query.state.data ? false : 3000),
    staleTime: Infinity,
  });

  const { data: workers = [] } = useQuery({
    queryKey: ["pipeline", runId, "stage2-workers", subFlowRunId],
    queryFn: () => fetchStage2Workers(subFlowRunId!),
    enabled: isActive && !!subFlowRunId,
    refetchInterval: 3000,
    staleTime: 1000,
  });

  const { data: logs = [] } = useQuery({
    queryKey: ["pipeline", runId, "stage2-logs", subFlowRunId],
    queryFn: () => fetchStage2Logs(subFlowRunId!),
    enabled: isActive && !!subFlowRunId,
    refetchInterval: 3000,
    staleTime: 1000,
  });

  return { workers, logs, subFlowRunId: subFlowRunId ?? null };
}
