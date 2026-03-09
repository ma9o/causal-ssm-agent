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
  subFlowRunId: string | null;
}

interface PrefectFlowRun {
  id: string;
  name: string;
  state_type: string;
}

interface PrefectTaskRun {
  id: string;
  name: string;
  state_type: string;
}

async function findStage2FlowRunId(): Promise<string | null> {
  const res = await fetch("/prefect/flow_runs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      flows: { name: { any_: ["stage2-worker-extraction"] } },
      sort: "START_TIME_DESC",
      limit: 1,
    }),
  });
  if (!res.ok) return null;
  const flowRuns: PrefectFlowRun[] = await res.json();
  return flowRuns.length > 0 ? flowRuns[0].id : null;
}

async function fetchStage2Workers(
  subFlowRunId: string,
): Promise<Stage2Worker[]> {
  const res = await fetch("/prefect/task_runs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      flow_runs: { id: { any_: [subFlowRunId] } },
      task_runs: { name: { startswith_: ["extract-chunk-"] } },
      sort: "EXPECTED_START_TIME_ASC",
    }),
  });
  if (!res.ok) return [];
  const taskRuns: PrefectTaskRun[] = await res.json();
  return taskRuns.map((tr) => ({
    id: tr.id,
    name: tr.name,
    state: mapState(tr.state_type),
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
    queryFn: findStage2FlowRunId,
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
