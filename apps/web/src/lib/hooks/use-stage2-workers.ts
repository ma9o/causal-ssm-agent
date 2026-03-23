"use client";

import type { StageRunStatus } from "./use-run-events";
import type {
  PrefectLogEntry,
  PrefectLogsResult,
} from "./use-stage-logs";
import { useStageLogs } from "./use-stage-logs";
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
  logBootstrapStatus: PrefectLogsResult["bootstrapStatus"];
  logConnectionState: PrefectLogsResult["connectionState"];
}

const STAGE2_LOG_PAGE_SIZE = 500;

export function getStage2WorkerQueryKey(
  workspaceId: string,
  rootFlowRunId: string | null,
) {
  return ["pipeline", workspaceId, "stage2-workers", rootFlowRunId] as const;
}

export function getStage2WorkerQueryKeyPrefix(workspaceId: string) {
  return ["pipeline", workspaceId, "stage2-workers"] as const;
}

/**
 * Stage-2 worker progress via WebSocket events + bootstrap/backfill + Prefect live log streaming.
 *
 * Worker states (submitted/completed/failed) arrive over the existing
 * WebSocket connection in use-run-events.ts and are written into the
 * ["pipeline", workspaceId, "stage2-workers", rootFlowRunId] query cache key.
 *
 * Logs are bootstrapped via REST once and then appended from Prefect's logs/out socket.
 */
export function useStage2Workers(
  workspaceId: string,
  rootFlowRunId: string | null,
  stageSubflowRunId: string | null,
  initialLogFlowRunIds: string[],
  stageStatus: StageRunStatus,
): Stage2WorkerProgress {
  const isActive = stageStatus === "running";

  // Workers: populated by WebSocket events in use-run-events.ts
  const { data: workers = [] } = useQuery<Stage2Worker[]>({
    queryKey: getStage2WorkerQueryKey(workspaceId, rootFlowRunId),
    queryFn: () => [],
    enabled: isActive && !!rootFlowRunId,
    staleTime: Infinity,
  });

  const {
    logs,
    bootstrapStatus: logBootstrapStatus,
    connectionState: logConnectionState,
  } = useStageLogs(workspaceId, "stage-2", stageSubflowRunId, initialLogFlowRunIds, stageStatus, {
    pageSize: STAGE2_LOG_PAGE_SIZE,
  });

  return { workers, logs, logBootstrapStatus, logConnectionState };
}
