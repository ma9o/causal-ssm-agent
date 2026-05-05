"use client";

import type { AnalysisStageRun } from "@/lib/api/analysis";
import { getPrefectLogsUrl } from "@/lib/runtime-urls";
import type { StageId } from "@causal-ssm/api-types";
import { getStageLogQueryScopeKey } from "@/lib/stage-observability";
import {
  buildPrefectLogStreamFilterMessage,
  fetchIncrementalPrefectLogs,
  getPrefectLogPageSize,
  mergePrefectLogs,
  type PrefectLogTimeWindow,
  type PrefectLogEntry,
} from "@/lib/prefect-log-client";
import { type QueryStatus, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import type { StageRunStatus } from "./pipeline-progress";
import {
  type PrefectSocketConnectionState,
  usePrefectSocketSubscription,
} from "./use-prefect-socket";
import { useStageLogScope } from "./use-stage-log-scope";

export interface PrefectLogsResult {
  logs: PrefectLogEntry[];
  bootstrapStatus: QueryStatus;
  connectionState: PrefectSocketConnectionState;
}

interface PrefectLogSocketMessage {
  type?: string;
  log?: PrefectLogEntry;
}

export interface PrefectLogStreamTransportArgs {
  enabled: boolean;
  flowRunIds: string[];
  timeWindow: PrefectLogTimeWindow;
  subscriptionKey: string;
  onLog: (log: PrefectLogEntry) => void;
  onSubscribed: () => void;
}

export interface PrefectLogTransport {
  fetchLogs: typeof fetchIncrementalPrefectLogs;
  useLogStream: (args: PrefectLogStreamTransportArgs) => PrefectSocketConnectionState;
}

function usePrefectWebSocketLogStream({
  enabled,
  flowRunIds,
  timeWindow,
  subscriptionKey,
  onLog,
  onSubscribed,
}: PrefectLogStreamTransportArgs) {
  const handleLogMessage = useCallback(
    (message: PrefectLogSocketMessage) => {
      if (message.type !== "log" || !message.log) {
        return;
      }

      onLog(message.log);
    },
    [onLog],
  );

  return usePrefectSocketSubscription<PrefectLogSocketMessage>({
    enabled,
    subscriptionKey,
    getSocketUrl: () => getPrefectLogsUrl(window.location.origin),
    buildFilterMessage: () =>
      buildPrefectLogStreamFilterMessage(flowRunIds, new Date(), timeWindow),
    onSubscribed,
    onMessage: handleLogMessage,
  });
}

export const prefectLogTransport: PrefectLogTransport = {
  fetchLogs: fetchIncrementalPrefectLogs,
  useLogStream: usePrefectWebSocketLogStream,
};

function usePrefectLogStream(
  queryKey: readonly unknown[],
  flowRunIds: string[],
  timeWindow: PrefectLogTimeWindow,
  subscriptionKey: string,
  enabled: boolean,
  transport: PrefectLogTransport,
) {
  const queryClient = useQueryClient();

  const handleLog = useCallback(
    (log: PrefectLogEntry) => {
      queryClient.setQueryData<PrefectLogEntry[]>(queryKey, (old) =>
        mergePrefectLogs(old ?? [], [log]),
      );
    },
    [queryClient, queryKey],
  );

  const handleSubscribed = useCallback(() => {
    queryClient.invalidateQueries({ queryKey });
  }, [queryClient, queryKey]);

  return transport.useLogStream({
    enabled,
    flowRunIds,
    timeWindow,
    subscriptionKey,
    onLog: handleLog,
    onSubscribed: handleSubscribed,
  });
}

export function usePrefectLogs(
  queryKey: readonly unknown[],
  flowRunIds: string[],
  timeWindow: PrefectLogTimeWindow,
  subscriptionKey: string,
  status: StageRunStatus,
  {
    pageSize = getPrefectLogPageSize(),
    transport = prefectLogTransport,
  }: {
    pageSize?: number;
    transport?: PrefectLogTransport;
  } = {},
): PrefectLogsResult {
  const queryClient = useQueryClient();
  const previousStatusRef = useRef<StageRunStatus>(status);
  const previousFlowRunIdsRef = useRef<string>(subscriptionKey);
  const previousBootstrapScopeRef = useRef<string>(subscriptionKey);

  const { data: logs = [], status: bootstrapStatus } = useQuery({
    queryKey,
    queryFn: async () => {
      const existing = queryClient.getQueryData<PrefectLogEntry[]>(queryKey) ?? [];
      const restartBootstrapFromBeginning = previousBootstrapScopeRef.current !== subscriptionKey;
      const nextLogs = await transport.fetchLogs(flowRunIds, existing, {
        limit: pageSize,
        offset: restartBootstrapFromBeginning ? 0 : existing.length,
        timeWindow,
      });
      previousBootstrapScopeRef.current = subscriptionKey;
      return nextLogs;
    },
    enabled: status !== "pending" && flowRunIds.length > 0,
    refetchInterval: false,
    staleTime: Infinity,
  });

  const connectionState = usePrefectLogStream(
    queryKey,
    flowRunIds,
    timeWindow,
    subscriptionKey,
    status === "running" && flowRunIds.length > 0 && bootstrapStatus === "success",
    transport,
  );

  useEffect(() => {
    if (flowRunIds.length === 0 || status === "pending") {
      previousFlowRunIdsRef.current = subscriptionKey;
      return;
    }

    if (previousFlowRunIdsRef.current !== subscriptionKey) {
      previousFlowRunIdsRef.current = subscriptionKey;
      queryClient.invalidateQueries({ queryKey });
    }
  }, [flowRunIds.length, queryClient, queryKey, status, subscriptionKey]);

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
  stageRun: AnalysisStageRun | null | undefined,
  status: StageRunStatus,
  {
    pageSize = getPrefectLogPageSize(),
    transport = prefectLogTransport,
  }: {
    pageSize?: number;
    transport?: PrefectLogTransport;
  } = {},
): PrefectLogsResult {
  const { runtime, flowRunIds, timeWindow, subscriptionKey } = useStageLogScope(
    workspaceId,
    stageId,
    stageRun,
    status,
  );
  const queryKey = [
    "pipeline",
    workspaceId,
    "logs",
    stageId,
    getStageLogQueryScopeKey(runtime),
  ] as const;
  return usePrefectLogs(queryKey, flowRunIds, timeWindow, subscriptionKey, status, {
    pageSize,
    transport,
  });
}
