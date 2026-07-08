import type { AnalysisTransitionRun } from "@/lib/api/analysis";

export const TRANSITION_EVENT_PREFIX = "nof1-causal-lab.transition";
export const TRANSITION_EVENT_FILTER_PREFIX = `${TRANSITION_EVENT_PREFIX}.`;

export type TransitionProgressStatus = "running" | "completed" | "failed";
export type TransitionObservedStatus = "pending" | TransitionProgressStatus;

const TRANSITION_STATUS_PRIORITY: Record<TransitionObservedStatus, number> = {
  pending: 0,
  running: 1,
  completed: 2,
  failed: 2,
};

export function getTransitionRunStatus(
  transitionRun: Pick<AnalysisTransitionRun, "execution"> | null | undefined,
): TransitionProgressStatus | null {
  const stateType = transitionRun?.execution?.stateType;
  if (!stateType) {
    return null;
  }

  switch (stateType.toUpperCase()) {
    case "RUNNING":
      return "running";
    case "COMPLETED":
      return "completed";
    case "FAILED":
      return "failed";
    default:
      return null;
  }
}

export function resolveTransitionObservedStatus(
  currentStatus: TransitionObservedStatus,
  transitionRun: Pick<AnalysisTransitionRun, "execution"> | null | undefined,
): TransitionObservedStatus {
  const transitionRunStatus = getTransitionRunStatus(transitionRun);
  if (!transitionRunStatus) {
    return currentStatus;
  }

  const currentPriority = TRANSITION_STATUS_PRIORITY[currentStatus];
  const transitionRunPriority = TRANSITION_STATUS_PRIORITY[transitionRunStatus];

  if (transitionRunPriority > currentPriority) {
    return transitionRunStatus;
  }

  if (
    transitionRunPriority === currentPriority &&
    currentStatus === "completed" &&
    transitionRunStatus === "failed"
  ) {
    return "failed";
  }

  return currentStatus;
}
