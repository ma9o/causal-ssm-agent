import type { AnalysisStageRun } from "@/lib/api/analysis";

export const STAGE_PROGRESS_EVENT_PREFIX = "nof1-causal-lab.pipeline-stage";
export const STAGE_PROGRESS_EVENT_FILTER_PREFIX = `${STAGE_PROGRESS_EVENT_PREFIX}.`;

export type StageProgressStatus = "running" | "completed" | "failed";
export type StageObservedStatus = "pending" | StageProgressStatus;

const STAGE_STATUS_PRIORITY: Record<StageObservedStatus, number> = {
  pending: 0,
  running: 1,
  completed: 2,
  failed: 2,
};

export function getStageRunStatus(
  stageRun: Pick<AnalysisStageRun, "execution"> | null | undefined,
): StageProgressStatus | null {
  const stateType = stageRun?.execution?.stateType;
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

export function resolveStageObservedStatus(
  currentStatus: StageObservedStatus,
  stageRun: Pick<AnalysisStageRun, "execution"> | null | undefined,
): StageObservedStatus {
  const stageRunStatus = getStageRunStatus(stageRun);
  if (!stageRunStatus) {
    return currentStatus;
  }

  const currentPriority = STAGE_STATUS_PRIORITY[currentStatus];
  const stageRunPriority = STAGE_STATUS_PRIORITY[stageRunStatus];

  if (stageRunPriority > currentPriority) {
    return stageRunStatus;
  }

  if (
    stageRunPriority === currentPriority &&
    currentStatus === "completed" &&
    stageRunStatus === "failed"
  ) {
    return "failed";
  }

  return currentStatus;
}
