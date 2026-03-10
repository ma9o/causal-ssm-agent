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

interface PrefectFlowRun {
  id: string;
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

export async function fetchStageFlowRunId(
  runId: string,
  stageId: StageId,
): Promise<string | null> {
  const stage = STAGES.find((s) => s.id === stageId);
  if (!stage) return null;

  const parentTaskRunsRes = await fetch("/prefect/task_runs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      flow_runs: { id: { any_: [runId] } },
      sort: "EXPECTED_START_TIME_DESC",
    }),
  });
  if (!parentTaskRunsRes.ok) return null;

  const taskRuns: PrefectTaskRun[] = await parentTaskRunsRes.json();
  const parentTaskRun = taskRuns.find(
    (candidate) => candidate.name === stage.prefectFlowName,
  );
  if (!parentTaskRun) return null;

  const flowRunsRes = await fetch("/prefect/flow_runs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      flows: { name: { any_: [stage.prefectFlowName] } },
      flow_runs: { parent_task_run_id: { any_: [parentTaskRun.id] } },
      sort: "START_TIME_DESC",
      limit: 1,
    }),
  });
  if (!flowRunsRes.ok) return null;

  const flowRuns: PrefectFlowRun[] = await flowRunsRes.json();
  return flowRuns[0]?.id ?? null;
}

async function fetchLogs(
  stageFlowRunId: string | null,
): Promise<PrefectLogEntry[]> {
  if (!stageFlowRunId) return [];

  const res = await fetch("/prefect/logs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      logs: { flow_run_id: { any_: [stageFlowRunId] } },
      sort: "TIMESTAMP_ASC",
      limit: 200,
    }),
  });
  if (!res.ok) return [];
  return res.json();
}

export function useStageLogs(runId: string, stageId: StageId, status: StageRunStatus) {
  const isActive = status !== "pending";

  const { data: stageFlowRunId } = useQuery({
    queryKey: ["pipeline", runId, "stageFlowRunId", stageId],
    queryFn: () => fetchStageFlowRunId(runId, stageId),
    enabled: isActive,
    staleTime: Infinity,
  });

  const { data: logs = [] } = useQuery({
    queryKey: ["pipeline", runId, "logs", stageId, stageFlowRunId],
    queryFn: () => fetchLogs(stageFlowRunId ?? null),
    enabled: isActive && stageFlowRunId !== undefined,
    refetchInterval: status === "running" ? 3000 : false,
    staleTime: status === "running" ? 1000 : Infinity,
  });

  return logs;
}
