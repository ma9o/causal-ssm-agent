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

interface StageLogSourcesResponse {
  logFlowRunIds: string[];
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

export function buildStageLogSourcesPath(
  userId: string,
  stageId: StageId,
  stageSubflowRunId: string,
) {
  const search = new URLSearchParams({
    stageId,
    stageSubflowRunId,
  });
  return `/api/analysis/${userId}/stage-log-sources?${search.toString()}`;
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

function useStageLogFlowRunIds(
  userId: string,
  stageId: StageId,
  stageSubflowRunId: string | null,
  initialLogFlowRunIds: string[],
  status: StageRunStatus,
) {
  const initialFlowRunIds =
    initialLogFlowRunIds.length > 0
      ? initialLogFlowRunIds
      : stageSubflowRunId
        ? [stageSubflowRunId]
        : [];

  const initialSignature = initialFlowRunIds.join("|");
  const isRefreshingStage2Sources = stageId === "stage-2" && status === "running" && !!stageSubflowRunId;

  const { data } = useQuery({
    queryKey: [
      "analysis",
      userId,
      "stage-log-sources",
      stageId,
      stageSubflowRunId,
      initialSignature,
    ] as const,
    queryFn: async () => {
      const response = await fetch(
        buildStageLogSourcesPath(userId, stageId, stageSubflowRunId as string),
        { cache: "no-store" },
      );
      if (!response.ok) {
        return initialFlowRunIds;
      }
      const payload = (await response.json()) as StageLogSourcesResponse;
      return payload.logFlowRunIds;
    },
    enabled: isRefreshingStage2Sources,
    initialData: initialFlowRunIds,
    initialDataUpdatedAt: 0,
    refetchInterval: 3000,
    staleTime: 1000,
  });

  return isRefreshingStage2Sources ? (data ?? initialFlowRunIds) : initialFlowRunIds;
}

function usePrefectLogStream(
  queryKey: readonly unknown[],
  flowRunIds: string[],
  enabled: boolean,
) {
  const queryClient = useQueryClient();
  const queryKeyRef = useRef(queryKey);
  const flowRunIdsRef = useRef(flowRunIds);

  queryKeyRef.current = queryKey;
  flowRunIdsRef.current = flowRunIds;

  const handleLogMessage = useCallback(
    (message: PrefectLogSocketMessage) => {
      if (message.type !== "log" || !message.log) {
        return;
      }

      queryClient.setQueryData<PrefectLogEntry[]>(queryKeyRef.current, (old) =>
        mergePrefectLogs(old ?? [], [message.log as PrefectLogEntry]),
      );
    },
    [queryClient],
  );

  return usePrefectSocketSubscription<PrefectLogSocketMessage>({
    enabled,
    getSocketUrl: () => getPrefectLogsUrl(window.location.origin),
    buildFilterMessage: () => buildPrefectLogStreamFilterMessage(flowRunIdsRef.current),
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
  const queryKeyRef = useRef(queryKey);

  queryKeyRef.current = queryKey;

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
      queryClient.invalidateQueries({ queryKey: queryKeyRef.current });
    }
  }, [flowRunIds.length, flowRunIdsSignature, queryClient, status]);

  useEffect(() => {
    const previousStatus = previousStatusRef.current;
    previousStatusRef.current = status;

    if (previousStatus === "running" && status !== "running" && flowRunIds.length > 0) {
      queryClient.invalidateQueries({ queryKey: queryKeyRef.current });
    }
  }, [flowRunIds.length, queryClient, status]);

  return {
    logs,
    bootstrapStatus,
    connectionState,
  };
}

export function useStageLogs(
  userId: string,
  stageId: StageId,
  stageSubflowRunId: string | null,
  initialLogFlowRunIds: string[],
  status: StageRunStatus,
  {
    pageSize = LOG_PAGE_SIZE,
  }: {
    pageSize?: number;
  } = {},
): PrefectLogsResult {
  const flowRunIds = useStageLogFlowRunIds(
    userId,
    stageId,
    stageSubflowRunId,
    initialLogFlowRunIds,
    status,
  );
  const queryKey = ["pipeline", userId, "logs", stageId, stageSubflowRunId] as const;
  return usePrefectLogs(queryKey, flowRunIds, status, { pageSize });
}
