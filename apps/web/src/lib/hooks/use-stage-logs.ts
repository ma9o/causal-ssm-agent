"use client";

import type { StageId } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";
import { useQuery } from "@tanstack/react-query";
import type { StageRunStatus } from "./use-run-events";

export interface PrefectLogEntry {
  id: string;
  created: string;
  name: string;
  level: number;
  message: string;
  timestamp: string;
  flow_run_id: string;
  task_run_id: string | null;
}

interface PrefectTaskRun {
  id: string;
  name: string;
}

const LOG_LEVEL_LABELS: Record<number, string> = {
  10: "DEBUG",
  20: "INFO",
  30: "WARNING",
  40: "ERROR",
  50: "CRITICAL",
};

export function logLevelLabel(level: number): string {
  return LOG_LEVEL_LABELS[level] ?? `L${level}`;
}

async function fetchTaskRunId(runId: string, stageId: StageId): Promise<string | null> {
  const stage = STAGES.find((s) => s.id === stageId);
  if (!stage) return null;

  const res = await fetch("/prefect/task_runs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      flow_runs: { id: { any_: [runId] } },
      task_runs: { name: { startswith_: [stage.prefectTaskName] } },
    }),
  });
  if (!res.ok) return null;

  const taskRuns: PrefectTaskRun[] = await res.json();
  return taskRuns.length > 0 ? taskRuns[0].id : null;
}

async function fetchLogs(
  runId: string,
  taskRunId: string | null,
): Promise<PrefectLogEntry[]> {
  const filter: Record<string, unknown> = {
    flow_run_id: { any_: [runId] },
  };
  if (taskRunId) {
    filter.task_run_id = { any_: [taskRunId] };
  }

  const res = await fetch("/prefect/logs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      logs: filter,
      sort: "TIMESTAMP_ASC",
      limit: 500,
    }),
  });
  if (!res.ok) return [];
  return res.json();
}

export function useStageLogs(runId: string, stageId: StageId, status: StageRunStatus) {
  const isActive = status === "running" || status === "completed";

  const { data: taskRunId } = useQuery({
    queryKey: ["pipeline", runId, "taskRunId", stageId],
    queryFn: () => fetchTaskRunId(runId, stageId),
    enabled: isActive,
    staleTime: Infinity,
  });

  const { data: logs = [] } = useQuery({
    queryKey: ["pipeline", runId, "logs", stageId, taskRunId],
    queryFn: () => fetchLogs(runId, taskRunId ?? null),
    enabled: isActive && taskRunId !== undefined,
    refetchInterval: status === "running" ? 3000 : false,
    staleTime: status === "running" ? 1000 : Infinity,
  });

  return logs;
}
