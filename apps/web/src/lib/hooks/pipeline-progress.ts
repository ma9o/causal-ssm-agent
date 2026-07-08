import type { ArtifactViewId, ArtifactStatus } from "@nof1-causal-lab/api-types";
import type { StaleArtifactsByProducer } from "@/lib/artifact-staleness";

export type TransitionRunStatus = Exclude<ArtifactStatus, "blocked">;

export interface TransitionTiming {
  startedAt: number;
  completedAt?: number;
}

export interface PipelineProgress {
  artifacts: Record<ArtifactViewId, TransitionRunStatus>;
  timings: Partial<Record<ArtifactViewId, TransitionTiming>>;
  /** Failure detail per transition (raised transition / failed telemetry event). */
  transitionErrors: Partial<Record<ArtifactViewId, string>>;
  /** Backend-computed freshness report, grouped by producing artifact for display. */
  staleArtifactsByProducer: StaleArtifactsByProducer;
  /** Whether the facade's auto-run driver is currently active. */
  autoRunning: boolean;
  /** Artifact display order from the machine's topological artifact order. */
  transitionOrder: ArtifactViewId[];
  /** Currently running transitions; plural because independent branches can execute concurrently. */
  runningTransitions: ArtifactViewId[];
  isComplete: boolean;
  isFailed: boolean;
}

const TRANSITION_STATUS_PRIORITY: Record<TransitionRunStatus, number> = {
  pending: 0,
  running: 1,
  completed: 2,
  failed: 2,
};

function getRunningTransitions(
  artifacts: Record<ArtifactViewId, TransitionRunStatus>,
  transitionOrder: readonly ArtifactViewId[],
): ArtifactViewId[] {
  return transitionOrder.filter((artifactId) => artifacts[artifactId] === "running");
}

function requireTransitionOrder(
  prev: PipelineProgress | undefined,
  transitionOrder: readonly ArtifactViewId[] | undefined,
): readonly ArtifactViewId[] {
  const order = transitionOrder ?? prev?.transitionOrder;
  if (!order) {
    throw new Error("Transition progress requires machine topological artifact order");
  }
  return order;
}

function createPendingArtifacts(
  transitionOrder: readonly ArtifactViewId[],
): Record<ArtifactViewId, TransitionRunStatus> {
  const artifacts = {} as Record<ArtifactViewId, TransitionRunStatus>;
  for (const artifactId of transitionOrder) {
    artifacts[artifactId] = "pending";
  }
  return artifacts;
}

export function initialProgress(transitionOrder: readonly ArtifactViewId[]): PipelineProgress {
  const artifacts = createPendingArtifacts(transitionOrder);

  return {
    artifacts,
    timings: {},
    transitionErrors: {},
    staleArtifactsByProducer: {},
    autoRunning: false,
    transitionOrder: [...transitionOrder],
    runningTransitions: [],
    isComplete: false,
    isFailed: false,
  };
}

/**
 * A `running` telemetry event begins a new attempt: unlike applyTransitionUpdate
 * (which merges unordered signals by priority), the ordered event stream may
 * legitimately re-run a completed or failed transition after its inputs changed,
 * so a terminal state is reset rather than preserved.
 */
export function restartTransitionAttempt(
  prev: PipelineProgress | undefined,
  artifactId: ArtifactViewId,
  eventTime?: number,
  transitionOrder?: readonly ArtifactViewId[],
): PipelineProgress {
  const order = requireTransitionOrder(prev, transitionOrder);
  const current = prev ?? initialProgress(order);
  if (current.artifacts[artifactId] === "running") {
    return applyTransitionUpdate(current, artifactId, "running", eventTime, undefined, order);
  }

  const ts = eventTime ?? Date.now();
  const artifacts = { ...current.artifacts, [artifactId]: "running" as TransitionRunStatus };
  const transitionErrors = { ...current.transitionErrors };
  delete transitionErrors[artifactId];

  return {
    ...current,
    artifacts,
    timings: { ...current.timings, [artifactId]: { startedAt: ts } },
    transitionErrors,
    runningTransitions: getRunningTransitions(artifacts, order),
    isComplete: false,
    isFailed: order.some((transitionId) => artifacts[transitionId] === "failed"),
  };
}

export function applyTransitionUpdate(
  prev: PipelineProgress | undefined,
  artifactId: ArtifactViewId,
  status: TransitionRunStatus,
  eventTime?: number,
  errorMessage?: string,
  transitionOrder?: readonly ArtifactViewId[],
): PipelineProgress {
  const order = requireTransitionOrder(prev, transitionOrder);
  const current = prev ?? initialProgress(order);
  const previousStatus = current.artifacts[artifactId];

  // A lower-priority signal never clobbers a higher one: a stale pending/running
  // must not undo a terminal state (genuine re-runs arrive via restartTransitionAttempt).
  if (TRANSITION_STATUS_PRIORITY[status] < TRANSITION_STATUS_PRIORITY[previousStatus]) {
    return current;
  }
  // completed and failed are equal-priority terminal states; the latest one wins
  // so a transition that failed then succeeded (or vice versa) shows its most recent
  // outcome. Only ignore a terminal signal that predates the recorded one.
  if (
    TRANSITION_STATUS_PRIORITY[status] === TRANSITION_STATUS_PRIORITY[previousStatus] &&
    previousStatus !== status
  ) {
    const prevCompletedAt = current.timings[artifactId]?.completedAt;
    if (eventTime !== undefined && prevCompletedAt !== undefined && eventTime < prevCompletedAt) {
      return current;
    }
  }

  const artifacts = { ...current.artifacts, [artifactId]: status };
  const ts = eventTime ?? Date.now();
  const existingTiming = current.timings[artifactId];
  const timings = { ...current.timings };

  if (status === "running") {
    timings[artifactId] = {
      startedAt: existingTiming?.startedAt ?? ts,
      completedAt: existingTiming?.completedAt,
    };
  } else {
    timings[artifactId] = {
      startedAt: existingTiming?.startedAt ?? ts,
      completedAt: ts,
    };
  }

  const isComplete = order.every((transitionId) => artifacts[transitionId] === "completed");
  const hasFailedTransition = order.some((transitionId) => artifacts[transitionId] === "failed");

  const transitionErrors =
    status === "failed" && errorMessage
      ? { ...current.transitionErrors, [artifactId]: errorMessage }
      : current.transitionErrors;

  return {
    ...current,
    artifacts,
    timings,
    transitionErrors,
    runningTransitions: getRunningTransitions(artifacts, order),
    isComplete,
    isFailed: current.isFailed || hasFailedTransition,
  };
}
