"use client";

import { useQuery } from "@tanstack/react-query";
import type { StageRunStatus } from "./pipeline-progress";

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

async function fetchLogs(
  stageSubflowRunId: string | null,
): Promise<PrefectLogEntry[]> {
  if (!stageSubflowRunId) return [];

  const res = await fetch("/prefect/logs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      logs: { flow_run_id: { any_: [stageSubflowRunId] } },
      sort: "TIMESTAMP_ASC",
      limit: 200,
    }),
  });
  if (!res.ok) return [];
  return res.json();
}

export function useStageLogs(
  userId: string,
  stageSubflowRunId: string | null,
  status: StageRunStatus,
) {
  const isActive = status !== "pending";

  const { data: logs = [] } = useQuery({
    queryKey: ["pipeline", userId, "logs", stageSubflowRunId],
    queryFn: () => fetchLogs(stageSubflowRunId),
    enabled: isActive && !!stageSubflowRunId,
    refetchInterval: status === "running" ? 3000 : false,
    staleTime: status === "running" ? 1000 : Infinity,
  });

  return logs;
}
