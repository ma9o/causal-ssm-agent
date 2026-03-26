"use client";

import { getPrefectLogsUrl } from "@/lib/runtime-urls";
import type { StageId } from "@causal-ssm/api-types";
import { type QueryStatus, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import type { StageRunStatus } from "./pipeline-progress";
import {
  type PrefectSocketConnectionState,
  usePrefectSocketSubscription,
} from "./use-prefect-socket";

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

export interface PrefectLogsResult {
  logs: PrefectLogEntry[];
  bootstrapStatus: QueryStatus;
  connectionState: PrefectSocketConnectionState;
}

interface PrefectLogSocketMessage {
  type?: string;
  log?: PrefectLogEntry;
}

const LOG_PAGE_SIZE = 200;
const LOG_STREAM_LOOKBACK_MS = 60_000;
const LOG_STREAM_LOOKAHEAD_MS = 365 * 24 * 60 * 60 * 1000;

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

export function mergePrefectLogs(
  existing: PrefectLogEntry[],
  incoming: PrefectLogEntry[],
): PrefectLogEntry[] {
  if (incoming.length === 0) {
    return existing;
  }

  const seen = new Set(existing.map((entry) => entry.id));
  const appended = incoming.filter((entry) => {
    if (seen.has(entry.id)) {
      return false;
    }
    seen.add(entry.id);
    return true;
  });

  return appended.length > 0 ? [...existing, ...appended] : existing;
}

export function buildPrefectLogFilterBody(
  flowRunIds: string[],
  offset = 0,
  limit = LOG_PAGE_SIZE,
) {
  return {
    offset,
    logs: {
      flow_run_id: { any_: flowRunIds },
    },
    sort: "TIMESTAMP_ASC",
    limit,
  };
}

export function buildPrefectLogStreamFilterMessage(
  flowRunIds: string[],
  now = new Date(),
) {
  return {
    type: "filter",
    filter: {
      flow_run_id: { any_: flowRunIds },
      timestamp: {
        after_: new Date(now.getTime() - LOG_STREAM_LOOKBACK_MS).toISOString(),
        before_: new Date(now.getTime() + LOG_STREAM_LOOKAHEAD_MS).toISOString(),
      },
    },
  };
}

async function fetchPrefectLogPage(
  flowRunIds: string[],
  offset = 0,
  limit = LOG_PAGE_SIZE,
): Promise<PrefectLogEntry[]> {
  if (flowRunIds.length === 0) return [];

  const res = await fetch("/prefect/logs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPrefectLogFilterBody(flowRunIds, offset, limit)),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function fetchIncrementalPrefectLogs(
  flowRunIds: string[],
  existing: PrefectLogEntry[],
  {
    limit = LOG_PAGE_SIZE,
  }: {
    limit?: number;
  } = {},
): Promise<PrefectLogEntry[]> {
  let merged = existing;
  let offset = existing.length;

  while (true) {
    const nextPage = await fetchPrefectLogPage(flowRunIds, offset, limit);
    if (nextPage.length === 0) {
      break;
    }

    merged = mergePrefectLogs(merged, nextPage);
    offset += nextPage.length;

    if (nextPage.length < limit) {
      break;
    }
  }

  return merged;
}

function usePrefectLogStream(
  queryKey: readonly unknown[],
  flowRunIds: string[],
  enabled: boolean,
) {
  const queryClient = useQueryClient();

  const handleLogMessage = useCallback(
    (message: PrefectLogSocketMessage) => {
      if (message.type !== "log" || !message.log) {
        return;
      }

      queryClient.setQueryData<PrefectLogEntry[]>(queryKey, (old) =>
        mergePrefectLogs(old ?? [], [message.log as PrefectLogEntry]),
      );
    },
    [queryClient, queryKey],
  );

  return usePrefectSocketSubscription<PrefectLogSocketMessage>({
    enabled,
    getSocketUrl: () => getPrefectLogsUrl(window.location.origin),
    buildFilterMessage: () => buildPrefectLogStreamFilterMessage(flowRunIds),
    onMessage: handleLogMessage,
  });
}

export function usePrefectLogs(
  queryKey: readonly unknown[],
  flowRunIds: string[],
  status: StageRunStatus,
  {
    pageSize = LOG_PAGE_SIZE,
  }: {
    pageSize?: number;
  } = {},
): PrefectLogsResult {
  const queryClient = useQueryClient();
  const flowRunIdsSignature = flowRunIds.join("|");
  const previousStatusRef = useRef<StageRunStatus>(status);
  const previousFlowRunIdsRef = useRef<string>(flowRunIdsSignature);

  const { data: logs = [], status: bootstrapStatus } = useQuery({
    queryKey,
    queryFn: async () => {
      const existing = queryClient.getQueryData<PrefectLogEntry[]>(queryKey) ?? [];
      return fetchIncrementalPrefectLogs(flowRunIds, existing, {
        limit: pageSize,
      });
    },
    enabled: status !== "pending" && flowRunIds.length > 0,
    refetchInterval: false,
    staleTime: Infinity,
  });

  const connectionState = usePrefectLogStream(
    queryKey,
    flowRunIds,
    status === "running" && flowRunIds.length > 0 && bootstrapStatus === "success",
  );

  useEffect(() => {
    if (flowRunIds.length === 0 || status === "pending") {
      previousFlowRunIdsRef.current = flowRunIdsSignature;
      return;
    }

    if (previousFlowRunIdsRef.current !== flowRunIdsSignature) {
      previousFlowRunIdsRef.current = flowRunIdsSignature;
      queryClient.invalidateQueries({ queryKey });
    }
  }, [flowRunIds.length, flowRunIdsSignature, queryClient, queryKey, status]);

  useEffect(() => {
    const previousStatus = previousStatusRef.current;
    previousStatusRef.current = status;

    if (previousStatus === "running" && status !== "running" && flowRunIds.length > 0) {
      queryClient.invalidateQueries({ queryKey });
    }
  }, [flowRunIds.length, queryClient, queryKey, status]);

  return {
    logs,
    bootstrapStatus,
    connectionState,
  };
}

export function useStageLogs(
  workspaceId: string,
  stageId: StageId,
  logFlowRunIds: string[],
  status: StageRunStatus,
  {
    pageSize = LOG_PAGE_SIZE,
  }: {
    pageSize?: number;
  } = {},
): PrefectLogsResult {
  const queryKey = ["pipeline", workspaceId, "logs", stageId, logFlowRunIds.join("|")] as const;
  return usePrefectLogs(queryKey, logFlowRunIds, status, { pageSize });
}
