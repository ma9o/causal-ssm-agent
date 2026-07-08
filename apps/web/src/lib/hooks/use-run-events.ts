"use client";

import type { ArtifactViewId } from "@nof1-causal-lab/api-types";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import {
  type EpisodeEventRecord,
  type EpisodeProgressPayload,
  type EpisodeTransitionRecord,
  getEpisodeProgress,
} from "@/lib/api/analysis";
import { groupStaleArtifactsByProducer, hasStaleArtifacts } from "@/lib/artifact-staleness";
import {
  TRANSITION_EVENT_FILTER_PREFIX,
  type TransitionProgressStatus,
} from "@/lib/transition-runtime";
import {
  applyExtractionEvent,
  getExtractionStateQueryKey,
  parseExtractionEvent,
  type ExtractionReplayState,
} from "@/lib/extraction-runtime";
import {
  applyModelSpecAdmissionEvent,
  getModelSpecAdmissionStateQueryKey,
  parseModelSpecAdmissionEvent,
  type ModelSpecAdmissionReplayState,
} from "@/lib/model-spec-admission-runtime";
import { isMockMode, simulatePipelineEvents } from "../api/mock-provider";
import {
  applyTransitionUpdate,
  initialProgress,
  type PipelineProgress,
  restartTransitionAttempt,
  type TransitionRunStatus,
} from "./pipeline-progress";
import { getArtifactViewQueryKey } from "./use-artifact-view";

export type { PipelineProgress, TransitionRunStatus, TransitionTiming } from "./pipeline-progress";

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

function isArtifactViewId(value: unknown): value is ArtifactViewId {
  return typeof value === "string" && TRANSITIONS.some((transition) => transition.id === value);
}

function isTransitionRunStatus(value: unknown): value is TransitionProgressStatus {
  return value === "running" || value === "completed" || value === "failed";
}

export interface TransitionProgressEvent {
  artifactId: ArtifactViewId;
  status: TransitionProgressStatus;
  eventTime?: number;
  error?: { type: string; message: string };
}

export function parseTransitionProgressEvent(
  record: EpisodeEventRecord,
): TransitionProgressEvent | null {
  if (!record.event.startsWith(TRANSITION_EVENT_FILTER_PREFIX)) {
    return null;
  }

  const payload = record.payload;
  const artifactId = payload?.transition_id;
  const status = payload?.status;
  if (!isArtifactViewId(artifactId) || !isTransitionRunStatus(status)) {
    return null;
  }

  return {
    artifactId,
    status,
    eventTime: cursorTimestampMs(record.cursor),
    error:
      payload?.error && typeof payload.error === "object"
        ? (payload.error as { type: string; message: string })
        : undefined,
  };
}

/** Adapt an episode event to the {event, occurred, payload} record telemetry parsers consume. */
function toRuntimeEventRecord(record: EpisodeEventRecord) {
  const timestampMs = cursorTimestampMs(record.cursor);
  return {
    event: record.event,
    occurred: timestampMs === undefined ? null : new Date(timestampMs).toISOString(),
    payload: record.payload,
  };
}

function invalidateArtifactView(
  queryClient: ReturnType<typeof useQueryClient>,
  workspaceId: string,
  artifactId: ArtifactViewId,
) {
  queryClient.invalidateQueries({ queryKey: getArtifactViewQueryKey(workspaceId, artifactId) });
}

/**
 * The durable journal is authoritative for a transition's terminal state: an applied
 * run transition means it completed, a raised one means it failed. Telemetry
 * `completed` events also drive completion, but they are ephemeral — the
 * transition keeps the display correct even when the event log has been pruned.
 */
function applyRunTransition(
  progress: PipelineProgress | undefined,
  transition: EpisodeTransitionRecord,
  transitionOrder: readonly ArtifactViewId[],
): PipelineProgress | undefined {
  if (transition.move.kind !== "run") {
    return progress;
  }
  const artifactId = transition.move.artifact_id;
  if (!isArtifactViewId(artifactId)) {
    return progress;
  }
  const eventTime = Date.parse(transition.ts);
  const ts = Number.isFinite(eventTime) ? eventTime : undefined;

  if (transition.status === "applied") {
    return applyTransitionUpdate(progress, artifactId, "completed", ts, undefined, transitionOrder);
  }
  if (transition.status === "raised") {
    return applyTransitionUpdate(
      progress,
      artifactId,
      "failed",
      ts,
      transition.error_message ?? transition.error_type ?? undefined,
      transitionOrder,
    );
  }
  return progress; // rejected attempts never executed — leave status untouched
}

function hasRunningTransition(progress: PipelineProgress | undefined): boolean {
  return (
    !!progress &&
    progress.transitionOrder.some((transitionId) => progress.artifacts[transitionId] === "running")
  );
}

function applyExistingArtifactView(
  progress: PipelineProgress | undefined,
  artifactId: ArtifactViewId,
  transitionOrder: readonly ArtifactViewId[],
): PipelineProgress {
  const current = progress ?? initialProgress(transitionOrder);
  if (current.artifacts[artifactId] !== "pending") {
    return current;
  }
  return applyTransitionUpdate(
    current,
    artifactId,
    "completed",
    undefined,
    undefined,
    transitionOrder,
  );
}

export function useRunEvents(
  workspaceId: string | null,
  transitionOrder: readonly ArtifactViewId[] | undefined,
) {
  const queryClient = useQueryClient();
  const cursorRef = useRef<string | null>(null);
  const lastSeqRef = useRef(0);
  const hydratedWorkspaceRef = useRef<string | null>(null);

  const updateTransition = useCallback(
    (
      artifactId: ArtifactViewId,
      status: TransitionRunStatus,
      eventTime?: number,
      errorMessage?: string,
    ) => {
      if (!transitionOrder) {
        return;
      }
      queryClient.setQueryData<PipelineProgress>(["pipeline", workspaceId, "status"], (old) =>
        applyTransitionUpdate(old, artifactId, status, eventTime, errorMessage, transitionOrder),
      );
    },
    [queryClient, transitionOrder, workspaceId],
  );

  const applyProgressPayload = useCallback(
    (payload: EpisodeProgressPayload) => {
      if (!workspaceId || !transitionOrder) {
        return;
      }

      for (const record of payload.events) {
        const runtimeRecord = toRuntimeEventRecord(record);

        const admissionEvent = parseModelSpecAdmissionEvent(runtimeRecord);
        if (admissionEvent) {
          queryClient.setQueryData<ModelSpecAdmissionReplayState>(
            getModelSpecAdmissionStateQueryKey(workspaceId),
            (old) => applyModelSpecAdmissionEvent(old, admissionEvent),
          );
          continue;
        }

        const extractionEvent = parseExtractionEvent(runtimeRecord);
        if (extractionEvent) {
          queryClient.setQueryData<ExtractionReplayState>(
            getExtractionStateQueryKey(workspaceId),
            (old) => applyExtractionEvent(old, extractionEvent),
          );
          continue;
        }

        const transitionEvent = parseTransitionProgressEvent(record);
        if (!transitionEvent) {
          continue;
        }

        if (transitionEvent.status === "running") {
          // The event stream is totally ordered, so a running event after a
          // terminal state is a genuine re-run (stale inputs recomputed).
          queryClient.setQueryData<PipelineProgress>(
            getPipelineStatusQueryKey(workspaceId),
            (old) =>
              restartTransitionAttempt(
                old,
                transitionEvent.artifactId,
                transitionEvent.eventTime,
                transitionOrder,
              ),
          );
          continue;
        }

        updateTransition(
          transitionEvent.artifactId,
          transitionEvent.status,
          transitionEvent.eventTime,
          transitionEvent.error?.message,
        );
        if (transitionEvent.status === "completed") {
          invalidateArtifactView(queryClient, workspaceId, transitionEvent.artifactId);
        }
      }
      if (payload.events.length > 0) {
        cursorRef.current = payload.events[payload.events.length - 1].cursor;
      }

      for (const artifact of payload.artifacts) {
        const artifactId = artifact.artifact_id;
        if (!artifact.exists || !isArtifactViewId(artifactId)) {
          continue;
        }
        queryClient.setQueryData<PipelineProgress>(getPipelineStatusQueryKey(workspaceId), (old) =>
          applyExistingArtifactView(old, artifactId, transitionOrder),
        );
      }

      for (const transition of payload.transitions) {
        if (transition.seq <= lastSeqRef.current) {
          continue;
        }
        queryClient.setQueryData<PipelineProgress>(
          getPipelineStatusQueryKey(workspaceId),
          (old) => applyRunTransition(old, transition, transitionOrder) ?? old,
        );
        lastSeqRef.current = Math.max(lastSeqRef.current, transition.seq);
      }

      queryClient.setQueryData<PipelineProgress>(getPipelineStatusQueryKey(workspaceId), (old) => ({
        ...(old ?? initialProgress(transitionOrder)),
        staleArtifactsByProducer: groupStaleArtifactsByProducer(payload.artifacts),
        autoRunning: payload.autoRunning,
      }));
    },
    [queryClient, transitionOrder, updateTransition, workspaceId],
  );

  // Reset the reduced caches when the workspace changes.
  useEffect(() => {
    if (!workspaceId || !transitionOrder || hydratedWorkspaceRef.current === workspaceId) {
      return;
    }
    hydratedWorkspaceRef.current = workspaceId;
    cursorRef.current = null;
    lastSeqRef.current = 0;

    queryClient.setQueryData(
      getPipelineStatusQueryKey(workspaceId),
      initialProgress(transitionOrder),
    );
    queryClient.removeQueries({ queryKey: getExtractionStateQueryKey(workspaceId) });
    queryClient.removeQueries({ queryKey: getModelSpecAdmissionStateQueryKey(workspaceId) });
  }, [queryClient, transitionOrder, workspaceId]);

  useEffect(() => {
    if (!workspaceId || !transitionOrder) return;

    if (isMockMode()) {
      const cleanup = simulatePipelineEvents(
        {
          onTransitionStart: (id) => updateTransition(id, "running"),
          onTransitionComplete: (id) => {
            updateTransition(id, "completed");
            invalidateArtifactView(queryClient, workspaceId, id);
          },
        },
        transitionOrder,
      );
      return () => {
        cleanup();
      };
    }
  }, [queryClient, transitionOrder, updateTransition, workspaceId]);

  useQuery({
    queryKey: getEpisodeProgressQueryKey(workspaceId ?? "__none__"),
    queryFn: async () => {
      const payload = await getEpisodeProgress(workspaceId as string, cursorRef.current);
      applyProgressPayload(payload);
      return payload;
    },
    // Read-only viewers poll too: published workspaces carry a real journal,
    // and a live local run publishing to the hosted store tails through here.
    enabled: !isMockMode() && !!workspaceId && !!transitionOrder,
    refetchInterval: (query) => {
      const payload = query.state.data;
      if (!payload) {
        return PROGRESS_POLL_INTERVAL_MS;
      }
      const progress = workspaceId
        ? queryClient.getQueryData<PipelineProgress>(getPipelineStatusQueryKey(workspaceId))
        : undefined;
      return payload.autoRunning ||
        hasRunningTransition(progress) ||
        hasStaleArtifacts(payload.artifacts)
        ? PROGRESS_POLL_INTERVAL_MS
        : false;
    },
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
}
