"use client";

import {
  getAnalysisManifestQueryKey,
  type AnalysisManifest,
  type AnalysisStageRuns,
  type AnalysisStageExecution,
} from "@/lib/api/analysis";
import { dedupeRootFlowRunIds, getLatestRootFlowRunId } from "@/lib/root-flow-runs";
import { getPrefectEventsUrl } from "@/lib/runtime-urls";
import {
  applyStage2Event,
  getStage2StateQueryKey,
  getStage2StateQueryKeyPrefix,
  parseStage2Event,
  type Stage2ReplayState,
} from "@/lib/stage2-runtime";
import {
  applyStage4Event,
  getStage4StateQueryKey,
  getStage4StateQueryKeyPrefix,
  parseStage4Event,
  type Stage4ReplayState,
} from "@/lib/stage4-runtime";
import {
  patchStageRun,
  normalizeLogFlowRunIds,
  normalizeStageSubflowRunId,
  STAGE_PROGRESS_EVENT_FILTER_PREFIX,
  type StageProgressStatus,
} from "@/lib/stage-runtime";
import type { StageId } from "@nof1-causal-lab/api-types";
import { STAGES } from "@nof1-causal-lab/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import { isMockMode, simulatePipelineEvents } from "../api/mock-provider";
import {
  applyStageUpdate,
  hasStoppedStage,
  initialProgress,
  mapPrefectTaskState,
  type PipelineProgress,
  type StageRunStatus,
} from "./pipeline-progress";
import { getStageDataQueryKey } from "./use-stage-data";
import { usePrefectSocketSubscription } from "./use-prefect-socket";

export type { PipelineProgress, StageRunStatus, StageTiming } from "./pipeline-progress";

const EVENT_LOOKBACK_MS = 60_000;
const EVENT_LOOKAHEAD_MS = 365 * 24 * 60 * 60 * 1000;
const NOF1_CAUSAL_LAB_EVENT_PREFIX = "nof1-causal-lab.";

interface PrefectEventSocketMessage {
  type?: string;
  event?: {
    event?: string;
    occurred?: string;
    payload?: Record<string, unknown>;
  };
}

function getPipelineStatusQueryKey(workspaceId: string) {
  return ["pipeline", workspaceId, "status"] as const;
}

export function buildPrefectEventFilterMessage(rootFlowRunId: string, now = new Date()) {
  return {
    type: "filter",
    filter: {
      // Pipeline emits custom events (stage progress + worker progress)
      // on the root flow run resource.
      event: { prefix: [NOF1_CAUSAL_LAB_EVENT_PREFIX] },
      resource: {
        id: [`prefect.flow-run.${rootFlowRunId}`],
      },
      // Prefect's websocket filter defaults `occurred.until` to "now".
      // Without an explicit future upper bound, the socket only backfills
      // historical events and drops all subsequent live task transitions.
      occurred: {
        since: new Date(now.getTime() - EVENT_LOOKBACK_MS).toISOString(),
        until: new Date(now.getTime() + EVENT_LOOKAHEAD_MS).toISOString(),
      },
    },
  };
}

function isStageId(value: unknown): value is StageId {
  return typeof value === "string" && STAGES.some((stage) => stage.id === value);
}

function isStageRunStatus(value: unknown): value is StageProgressStatus {
  return value === "running" || value === "completed" || value === "failed";
}

export interface StageProgressEvent {
  stageId: StageId;
  status: StageProgressStatus;
  eventTime?: number;
  occurred?: string;
  outcome?: string;
  error?: { type: string; message: string };
  stageSubflowRunId?: string;
  logFlowRunIds?: string[];
}

export function parsePrefectStageProgressEvent(
  event: PrefectEventSocketMessage["event"],
): StageProgressEvent | null {
  if (!event?.event?.startsWith(STAGE_PROGRESS_EVENT_FILTER_PREFIX)) {
    return null;
  }

  const payload = event.payload;
  const stageId = payload?.stage_id;
  const status = payload?.status;
  if (!isStageId(stageId) || !isStageRunStatus(status)) {
    return null;
  }

  const stageSubflowRunId = normalizeStageSubflowRunId(payload?.stage_subflow_run_id) ?? undefined;
  const explicitLogFlowRunIds = normalizeLogFlowRunIds(payload?.log_flow_run_ids);

  return {
    stageId,
    status,
    eventTime: event.occurred ? new Date(event.occurred).getTime() : undefined,
    occurred: event.occurred,
    outcome: typeof payload?.outcome === "string" ? payload.outcome : undefined,
    error:
      payload?.error && typeof payload.error === "object"
        ? (payload.error as { type: string; message: string })
        : undefined,
    stageSubflowRunId,
    logFlowRunIds: explicitLogFlowRunIds.length > 0 ? explicitLogFlowRunIds : undefined,
  };
}

function invalidateStageData(
  queryClient: ReturnType<typeof useQueryClient>,
  workspaceId: string,
  stageId: StageId,
) {
  queryClient.invalidateQueries({ queryKey: getStageDataQueryKey(workspaceId, stageId) });
}

function applyHydratedExecutionToProgress(
  progress: PipelineProgress,
  stageId: StageId,
  execution: AnalysisStageExecution,
): PipelineProgress {
  const status = mapPrefectTaskState(execution.stateType);
  if (!status) return progress;

  const startTime = execution.startTime ? new Date(execution.startTime).getTime() : undefined;
  const endTime = execution.endTime ? new Date(execution.endTime).getTime() : undefined;

  if (status === "completed") {
    let next = progress;
    if (startTime) {
      next = applyStageUpdate(next, stageId, "running", startTime);
    }
    return applyStageUpdate(next, stageId, "completed", endTime ?? startTime);
  }

  if (status === "running") {
    return applyStageUpdate(progress, stageId, "running", startTime);
  }

  let next = progress;
  if (startTime) {
    next = applyStageUpdate(next, stageId, "running", startTime);
  }
  return applyStageUpdate(next, stageId, "failed", endTime ?? startTime);
}

function isPipelineTerminal(
  queryClient: ReturnType<typeof useQueryClient>,
  workspaceId: string,
): boolean {
  const progress = queryClient.getQueryData<PipelineProgress>(
    getPipelineStatusQueryKey(workspaceId),
  );
  return progress?.isComplete === true || hasStoppedStage(progress) || progress?.isFailed === true;
}

function mergeHydratedManifestProgress(
  workspaceId: string,
  stageRuns: AnalysisStageRuns,
  queryClient: ReturnType<typeof useQueryClient>,
  progress?: PipelineProgress,
) {
  let next = progress ?? initialProgress();

  for (const stage of STAGES) {
    const execution = stageRuns[stage.id]?.execution;
    if (!execution) {
      continue;
    }

    next = applyHydratedExecutionToProgress(next, stage.id, execution);
    if (next.stages[stage.id] === "completed") {
      invalidateStageData(queryClient, workspaceId, stage.id);
    }
  }

  return next;
}

export function useRunEvents(
  workspaceId: string | null,
  rootFlowRunIds: string[],
  stageRuns?: AnalysisStageRuns,
) {
  const queryClient = useQueryClient();
  const activeRootFlowRunId = getLatestRootFlowRunId(rootFlowRunIds);
  const hydratedLineageKeyRef = useRef<string | null>(null);

  const updateStage = useCallback(
    (stageId: StageId, status: StageRunStatus, eventTime?: number, outcome?: string) => {
      queryClient.setQueryData<PipelineProgress>(["pipeline", workspaceId, "status"], (old) =>
        applyStageUpdate(old, stageId, status, eventTime, outcome),
      );
    },
    [queryClient, workspaceId],
  );

  const handlePrefectEventMessage = useCallback(
    (message: PrefectEventSocketMessage, socket: { close: () => void }) => {
      if (!workspaceId || !activeRootFlowRunId) {
        return;
      }

      const stage4Event = parseStage4Event(message.event);
      if (stage4Event) {
        queryClient.setQueryData<Stage4ReplayState>(
          getStage4StateQueryKey(workspaceId, activeRootFlowRunId),
          (old) => applyStage4Event(old, stage4Event),
        );
        return;
      }

      const stage2Event = parseStage2Event(message.event);
      if (stage2Event) {
        queryClient.setQueryData<Stage2ReplayState>(
          getStage2StateQueryKey(workspaceId, activeRootFlowRunId),
          (old) => applyStage2Event(old, stage2Event),
        );
        return;
      }

      const stageEvent = parsePrefectStageProgressEvent(message.event);
      if (!stageEvent) {
        return;
      }

      queryClient.setQueriesData<AnalysisManifest>(
        { queryKey: getAnalysisManifestQueryKey(workspaceId) },
        (old) =>
          old
            ? {
                ...old,
                stages: {
                  ...old.stages,
                  [stageEvent.stageId]: patchStageRun(
                    old.stages[stageEvent.stageId],
                    activeRootFlowRunId,
                    stageEvent,
                  ),
                },
              }
            : old,
      );
      updateStage(stageEvent.stageId, stageEvent.status, stageEvent.eventTime, stageEvent.outcome);
      if (stageEvent.status === "completed") {
        invalidateStageData(queryClient, workspaceId, stageEvent.stageId);
      }

      if (isPipelineTerminal(queryClient, workspaceId)) {
        socket.close();
      }
    },
    [activeRootFlowRunId, queryClient, updateStage, workspaceId],
  );

  useEffect(() => {
    if (!workspaceId) return;
    const normalizedRootFlowRunIds = dedupeRootFlowRunIds(rootFlowRunIds);
    const lineageKey = `${workspaceId}:${normalizedRootFlowRunIds.join("|")}`;

    if (hydratedLineageKeyRef.current === lineageKey) {
      return;
    }
    hydratedLineageKeyRef.current = lineageKey;

    // Initialize progress, then hydrate from Prefect to catch up on
    // stages that completed before this page loaded (session resumption).
    queryClient.setQueryData(getPipelineStatusQueryKey(workspaceId), initialProgress());
    queryClient.removeQueries({ queryKey: getStage2StateQueryKeyPrefix(workspaceId) });
    queryClient.removeQueries({ queryKey: getStage4StateQueryKeyPrefix(workspaceId) });
  }, [queryClient, rootFlowRunIds, workspaceId]);

  useEffect(() => {
    if (!workspaceId || !stageRuns) return;

    queryClient.setQueryData<PipelineProgress | undefined>(
      getPipelineStatusQueryKey(workspaceId),
      (old) => mergeHydratedManifestProgress(workspaceId, stageRuns, queryClient, old),
    );
  }, [queryClient, stageRuns, workspaceId]);

  useEffect(() => {
    if (!workspaceId) return;

    if (isMockMode()) {
      const cleanup = simulatePipelineEvents({
        onStageStart: (id) => updateStage(id, "running"),
        onStageComplete: (id) => {
          updateStage(id, "completed");
          invalidateStageData(queryClient, workspaceId, id);
        },
      });
      return () => {
        cleanup();
      };
    }
  }, [activeRootFlowRunId, queryClient, updateStage, workspaceId]);

  usePrefectSocketSubscription<PrefectEventSocketMessage>({
    enabled: !isMockMode() && !!workspaceId && !!activeRootFlowRunId,
    subscriptionKey: activeRootFlowRunId ?? "",
    getSocketUrl: () => getPrefectEventsUrl(window.location.origin),
    buildFilterMessage: () => buildPrefectEventFilterMessage(activeRootFlowRunId as string),
    onMessage: handlePrefectEventMessage,
  });
}
