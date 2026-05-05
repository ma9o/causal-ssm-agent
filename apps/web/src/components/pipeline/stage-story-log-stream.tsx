"use client";

import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import {
  usePrefectLogs,
  type PrefectLogStreamTransportArgs,
  type PrefectLogTransport,
} from "@/lib/hooks/use-stage-logs";
import type { PrefectSocketConnectionState } from "@/lib/hooks/use-prefect-socket";
import {
  getPrefectLogPageSize,
  mergePrefectLogs,
  type PrefectLogEntry,
  type PrefectLogTimeWindow,
} from "@/lib/prefect-log-client";
import { buildStageLogSubscriptionKey } from "@/lib/stage-observability";
import { useEffect, useEffectEvent, useId, useMemo, useState } from "react";
import { StageLogView } from "./stage-log-viewer";
import { createStageStoryLogs } from "./stage-story-log-fixture";

const STORY_FLOW_RUN_ID = "story-flow-run";
const DEFAULT_RUNNING_BOOTSTRAP_COUNT = 3;
const DEFAULT_STORY_LOG_INTERVAL_MS = 600;
const DEFAULT_CONNECT_DELAY_MS = 120;
const DEFAULT_AUTH_DELAY_MS = 120;

type StoryLogTransportOptions = {
  logs: PrefectLogEntry[];
  bootstrapCount: number;
  intervalMs?: number;
  connectDelayMs?: number;
  authDelayMs?: number;
};

function logMatchesScope(
  log: PrefectLogEntry,
  flowRunIds: readonly string[],
  timeWindow: PrefectLogTimeWindow,
): boolean {
  if (!flowRunIds.includes(log.flow_run_id)) {
    return false;
  }

  if (timeWindow.after && log.timestamp < timeWindow.after) {
    return false;
  }

  if (timeWindow.before && log.timestamp > timeWindow.before) {
    return false;
  }

  return true;
}

export function createStageStoryLogTransport({
  logs,
  bootstrapCount,
  intervalMs = DEFAULT_STORY_LOG_INTERVAL_MS,
  connectDelayMs = DEFAULT_CONNECT_DELAY_MS,
  authDelayMs = DEFAULT_AUTH_DELAY_MS,
}: StoryLogTransportOptions): PrefectLogTransport {
  let publishedCount = Math.min(Math.max(bootstrapCount, 0), logs.length);

  return {
    fetchLogs: async (
      flowRunIds,
      existing,
      { limit, offset = existing.length, timeWindow = {} } = {},
    ) => {
      const visibleLogs = logs
        .slice(0, publishedCount)
        .filter((log) => logMatchesScope(log, flowRunIds, timeWindow));
      const nextPage = visibleLogs.slice(offset, offset + (limit ?? getPrefectLogPageSize()));

      return mergePrefectLogs(existing, nextPage);
    },
    useLogStream({
      enabled,
      flowRunIds,
      timeWindow,
      subscriptionKey,
      onLog,
      onSubscribed,
    }: PrefectLogStreamTransportArgs): PrefectSocketConnectionState {
      const [connectionState, setConnectionState] = useState<PrefectSocketConnectionState>("idle");
      const emitLog = useEffectEvent((log: PrefectLogEntry) => onLog(log));
      const notifySubscribed = useEffectEvent(() => onSubscribed());

      useEffect(() => {
        if (!enabled) {
          setConnectionState("idle");
          return;
        }

        let disposed = false;
        let streamTimer: ReturnType<typeof setInterval> | undefined;
        const pendingTimers = new Set<ReturnType<typeof setTimeout>>();

        const schedule = (callback: () => void, delayMs: number) => {
          const timer = setTimeout(() => {
            pendingTimers.delete(timer);
            callback();
          }, delayMs);
          pendingTimers.add(timer);
        };

        const emitNextVisibleLog = () => {
          while (publishedCount < logs.length) {
            const nextLog = logs[publishedCount];
            publishedCount += 1;

            if (logMatchesScope(nextLog, flowRunIds, timeWindow)) {
              emitLog(nextLog);
              return true;
            }
          }

          return false;
        };

        setConnectionState("connecting");
        schedule(() => {
          if (disposed) return;
          setConnectionState("authenticating");
          schedule(() => {
            if (disposed) return;
            setConnectionState("streaming");
            notifySubscribed();

            streamTimer = setInterval(() => {
              if (!emitNextVisibleLog() && streamTimer) {
                clearInterval(streamTimer);
                streamTimer = undefined;
              }
            }, intervalMs);
          }, authDelayMs);
        }, connectDelayMs);

        return () => {
          disposed = true;
          setConnectionState("idle");
          for (const timer of pendingTimers) {
            clearTimeout(timer);
          }
          if (streamTimer) {
            clearInterval(streamTimer);
          }
        };
      }, [enabled, flowRunIds, subscriptionKey, timeWindow]);

      if (!enabled) {
        return "idle";
      }

      return connectionState === "idle" ? "connecting" : connectionState;
    },
  };
}

export function StoryStageLogView({
  status,
  storyId = "default",
  logs,
  logCount,
  bootstrapCount,
  intervalMs,
  pageSize = getPrefectLogPageSize(),
  flowRunId = STORY_FLOW_RUN_ID,
  timeWindow,
}: {
  status: StageRunStatus;
  storyId?: string;
  logs?: PrefectLogEntry[];
  logCount?: number;
  bootstrapCount?: number;
  intervalMs?: number;
  pageSize?: number;
  flowRunId?: string;
  timeWindow?: PrefectLogTimeWindow;
}) {
  const componentId = useId();
  const storyLogs = useMemo(() => {
    if (logs) {
      return logs;
    }

    return createStageStoryLogs(logCount).map((log) => ({
      ...log,
      flow_run_id: flowRunId,
    }));
  }, [flowRunId, logCount, logs]);
  const resolvedBootstrapCount =
    bootstrapCount ?? (status === "running" ? DEFAULT_RUNNING_BOOTSTRAP_COUNT : storyLogs.length);
  const resolvedTimeWindow = useMemo(() => timeWindow ?? {}, [timeWindow]);
  const flowRunIds = useMemo(() => [flowRunId], [flowRunId]);
  const subscriptionKey = useMemo(
    () => buildStageLogSubscriptionKey(flowRunIds, resolvedTimeWindow),
    [flowRunIds, resolvedTimeWindow],
  );
  const queryKey = useMemo(
    () => ["storybook", "stage-logs", storyId, componentId] as const,
    [componentId, storyId],
  );
  const transport = useMemo(
    () =>
      createStageStoryLogTransport({
        logs: storyLogs,
        bootstrapCount: resolvedBootstrapCount,
        intervalMs,
      }),
    [intervalMs, resolvedBootstrapCount, storyLogs],
  );

  const {
    logs: streamedLogs,
    bootstrapStatus,
    connectionState,
  } = usePrefectLogs(queryKey, flowRunIds, resolvedTimeWindow, subscriptionKey, status, {
    pageSize,
    transport,
  });

  return (
    <StageLogView
      logs={streamedLogs}
      status={status}
      bootstrapStatus={bootstrapStatus}
      connectionState={connectionState}
    />
  );
}
