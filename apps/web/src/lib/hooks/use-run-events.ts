"use client";

import {
  getEpisodeProgress,
  type EpisodeEventRecord,
  type EpisodeProgressPayload,
  type EpisodeTransitionRecord,
} from "@/lib/api/analysis";
import {
  applyStage2Event,
  getStage2StateQueryKey,
  parseStage2Event,
  type Stage2ReplayState,
} from "@/lib/stage2-runtime";
import {
  applyStage4Event,
  getStage4StateQueryKey,
  parseStage4Event,
  type Stage4ReplayState,
} from "@/lib/stage4-runtime";
import { groupStaleArtifactsByStage, hasStaleArtifacts } from "@/lib/artifact-staleness";
import { STAGE_PROGRESS_EVENT_FILTER_PREFIX, type StageProgressStatus } from "@/lib/stage-runtime";
import type { StageId } from "@nof1-causal-lab/api-types";
import { STAGES } from "@nof1-causal-lab/api-types";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import { isMockMode, simulatePipelineEvents } from "../api/mock-provider";
import {
  applyStageUpdate,
  initialProgress,
  restartStageAttempt,
  type PipelineProgress,
  type StageRunStatus,
} from "./pipeline-progress";
import { getStageDataQueryKey } from "./use-stage-data";

export type { PipelineProgress, StageRunStatus, StageTiming } from "./pipeline-progress";

const PROGRESS_POLL_INTERVAL_MS = 2_000;

function getPipelineStatusQueryKey(workspaceId: string) {
  return ["pipeline", workspaceId, "status"] as const;
}

export function getEpisodeProgressQueryKey(workspaceId: string) {
  return ["episode", workspaceId, "progress"] as const;
}

/** Event cursors are `{time_ns:020d}-{uuid}.json` filenames — time-ordered by construction. */
export function cursorTimestampMs(cursor: string): number | undefined {
  const nanos = cursor.split("-", 1)[0];
  if (!/^\d+$/.test(nanos)) {
    return undefined;
  }
  return Math.floor(Number(nanos) / 1_000_000);
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
  error?: { type: string; message: string };
}

export function parseStageProgressEvent(record: EpisodeEventRecord): StageProgressEvent | null {
  if (!record.event.startsWith(STAGE_PROGRESS_EVENT_FILTER_PREFIX)) {
    return null;
  }

  const payload = record.payload;
  const stageId = payload?.stage_id;
  const status = payload?.status;
  if (!isStageId(stageId) || !isStageRunStatus(status)) {
    return null;
  }

  return {
    stageId,
    status,
    eventTime: cursorTimestampMs(record.cursor),
    error:
      payload?.error && typeof payload.error === "object"
        ? (payload.error as { type: string; message: string })
        : undefined,
  };
}

/** Adapt an episode event to the {event, occurred, payload} record the stage parsers consume. */
function toRuntimeEventRecord(record: EpisodeEventRecord) {
  const timestampMs = cursorTimestampMs(record.cursor);
  return {
    event: record.event,
    occurred: timestampMs === undefined ? null : new Date(timestampMs).toISOString(),
    payload: record.payload,
  };
}

function invalidateStageData(
  queryClient: ReturnType<typeof useQueryClient>,
  workspaceId: string,
  stageId: StageId,
) {
  queryClient.invalidateQueries({ queryKey: getStageDataQueryKey(workspaceId, stageId) });
}

/**
 * The durable journal is authoritative for a stage's terminal state: an applied
 * run transition means it completed, a raised one means it failed. Telemetry
 * `completed` events also drive completion, but they are ephemeral — the
 * transition keeps the display correct even when the event log has been pruned.
 */
function applyRunTransition(
  progress: PipelineProgress | undefined,
  transition: EpisodeTransitionRecord,
): PipelineProgress | undefined {
  if (transition.move.kind !== "run") {
    return progress;
  }
  const stageId = transition.move.stage_id;
  if (!isStageId(stageId)) {
    return progress;
  }
  const eventTime = Date.parse(transition.ts);
  const ts = Number.isFinite(eventTime) ? eventTime : undefined;

  if (transition.status === "applied") {
    return applyStageUpdate(progress, stageId, "completed", ts);
  }
  if (transition.status === "raised") {
    return applyStageUpdate(
      progress,
      stageId,
      "failed",
      ts,
      transition.error_message ?? transition.error_type ?? undefined,
    );
  }
  return progress; // rejected attempts never executed — leave status untouched
}

function hasRunningStage(progress: PipelineProgress | undefined): boolean {
  return !!progress && STAGES.some((stage) => progress.stages[stage.id] === "running");
}

export function useRunEvents(workspaceId: string | null) {
  const queryClient = useQueryClient();
  const cursorRef = useRef<string | null>(null);
  const lastSeqRef = useRef(0);
  const hydratedWorkspaceRef = useRef<string | null>(null);

  const updateStage = useCallback(
    (stageId: StageId, status: StageRunStatus, eventTime?: number, errorMessage?: string) => {
      queryClient.setQueryData<PipelineProgress>(["pipeline", workspaceId, "status"], (old) =>
        applyStageUpdate(old, stageId, status, eventTime, errorMessage),
      );
    },
    [queryClient, workspaceId],
  );

  const applyProgressPayload = useCallback(
    (payload: EpisodeProgressPayload) => {
      if (!workspaceId) {
        return;
      }

      for (const record of payload.events) {
        const runtimeRecord = toRuntimeEventRecord(record);

        const stage4Event = parseStage4Event(runtimeRecord);
        if (stage4Event) {
          queryClient.setQueryData<Stage4ReplayState>(getStage4StateQueryKey(workspaceId), (old) =>
            applyStage4Event(old, stage4Event),
          );
          continue;
        }

        const stage2Event = parseStage2Event(runtimeRecord);
        if (stage2Event) {
          queryClient.setQueryData<Stage2ReplayState>(getStage2StateQueryKey(workspaceId), (old) =>
            applyStage2Event(old, stage2Event),
          );
          continue;
        }

        const stageEvent = parseStageProgressEvent(record);
        if (!stageEvent) {
          continue;
        }

        if (stageEvent.status === "running") {
          // The event stream is totally ordered, so a running event after a
          // terminal state is a genuine re-run (stale inputs recomputed).
          queryClient.setQueryData<PipelineProgress>(
            getPipelineStatusQueryKey(workspaceId),
            (old) => restartStageAttempt(old, stageEvent.stageId, stageEvent.eventTime),
          );
          continue;
        }

        updateStage(
          stageEvent.stageId,
          stageEvent.status,
          stageEvent.eventTime,
          stageEvent.error?.message,
        );
        if (stageEvent.status === "completed") {
          invalidateStageData(queryClient, workspaceId, stageEvent.stageId);
        }
      }
      if (payload.events.length > 0) {
        cursorRef.current = payload.events[payload.events.length - 1].cursor;
      }

      for (const transition of payload.transitions) {
        if (transition.seq <= lastSeqRef.current) {
          continue;
        }
        queryClient.setQueryData<PipelineProgress>(
          getPipelineStatusQueryKey(workspaceId),
          (old) => applyRunTransition(old, transition) ?? old,
        );
        lastSeqRef.current = Math.max(lastSeqRef.current, transition.seq);
      }

      queryClient.setQueryData<PipelineProgress>(getPipelineStatusQueryKey(workspaceId), (old) => ({
        ...(old ?? initialProgress()),
        staleArtifactsByStage: groupStaleArtifactsByStage(payload.artifacts),
        autoRunning: payload.autoRunning,
      }));
    },
    [queryClient, updateStage, workspaceId],
  );

  // Reset the reduced caches when the workspace changes.
  useEffect(() => {
    if (!workspaceId || hydratedWorkspaceRef.current === workspaceId) {
      return;
    }
    hydratedWorkspaceRef.current = workspaceId;
    cursorRef.current = null;
    lastSeqRef.current = 0;

    queryClient.setQueryData(getPipelineStatusQueryKey(workspaceId), initialProgress());
    queryClient.removeQueries({ queryKey: getStage2StateQueryKey(workspaceId) });
    queryClient.removeQueries({ queryKey: getStage4StateQueryKey(workspaceId) });
  }, [queryClient, workspaceId]);

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
  }, [queryClient, updateStage, workspaceId]);

  useQuery({
    queryKey: getEpisodeProgressQueryKey(workspaceId ?? "__none__"),
    queryFn: async () => {
      const payload = await getEpisodeProgress(workspaceId as string, cursorRef.current);
      applyProgressPayload(payload);
      return payload;
    },
    // Read-only viewers poll too: published workspaces carry a real journal,
    // and a live local run publishing to the hosted store tails through here.
    enabled: !isMockMode() && !!workspaceId,
    refetchInterval: (query) => {
      const payload = query.state.data;
      if (!payload) {
        return PROGRESS_POLL_INTERVAL_MS;
      }
      const progress = workspaceId
        ? queryClient.getQueryData<PipelineProgress>(getPipelineStatusQueryKey(workspaceId))
        : undefined;
      return payload.autoRunning ||
        hasRunningStage(progress) ||
        hasStaleArtifacts(payload.artifacts)
        ? PROGRESS_POLL_INTERVAL_MS
        : false;
    },
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
}
