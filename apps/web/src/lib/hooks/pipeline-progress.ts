import type { StageId, StageOutcome, StageStatus } from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";

export type StageRunStatus = Exclude<StageStatus, "blocked">;

export interface StageTiming {
  startedAt: number;
  completedAt?: number;
}

export interface PipelineProgress {
  stages: Record<StageId, StageRunStatus>;
  timings: Partial<Record<StageId, StageTiming>>;
  stageOutcomes: Partial<Record<StageId, StageOutcome>>;
  currentStage: StageId | null;
  isComplete: boolean;
  isFailed: boolean;
}

const STAGE_STATUS_PRIORITY: Record<StageRunStatus, number> = {
  pending: 0,
  running: 1,
  completed: 2,
  failed: 2,
};

function getCurrentRunningStage(stages: Record<StageId, StageRunStatus>): StageId | null {
  for (let i = STAGES.length - 1; i >= 0; i -= 1) {
    const stageId = STAGES[i]?.id;
    if (stageId && stages[stageId] === "running") {
      return stageId;
    }
  }
  return null;
}

export function initialProgress(): PipelineProgress {
  const stages = {} as Record<StageId, StageRunStatus>;
  for (const stage of STAGES) {
    stages[stage.id] = "pending";
  }

  return {
    stages,
    timings: {},
    stageOutcomes: {},
    currentStage: null,
    isComplete: false,
    isFailed: false,
  };
}

export function mapPrefectTaskState(
  stateType: string,
): StageRunStatus | null {
  switch (stateType.toUpperCase()) {
    case "RUNNING":
      return "running";
    case "COMPLETED":
      return "completed";
    case "FAILED":
    case "CRASHED":
    case "CANCELLED":
    case "CANCELLING":
      return "failed";
    default:
      return null;
  }
}

export function applyStageUpdate(
  prev: PipelineProgress | undefined,
  stageId: StageId,
  status: StageRunStatus,
  eventTime?: number,
): PipelineProgress {
  const current = prev ?? initialProgress();
  const previousStatus = current.stages[stageId];

  if (STAGE_STATUS_PRIORITY[status] < STAGE_STATUS_PRIORITY[previousStatus]) {
    return current;
  }
  if (
    STAGE_STATUS_PRIORITY[status] === STAGE_STATUS_PRIORITY[previousStatus] &&
    previousStatus !== status
  ) {
    return current;
  }

  const stages = { ...current.stages, [stageId]: status };
  const ts = eventTime ?? Date.now();
  const existingTiming = current.timings[stageId];
  const timings = { ...current.timings };

  if (status === "running") {
    timings[stageId] = {
      startedAt: existingTiming?.startedAt ?? ts,
      completedAt: existingTiming?.completedAt,
    };
  } else {
    timings[stageId] = {
      startedAt: existingTiming?.startedAt ?? ts,
      completedAt: ts,
    };
  }

  const isComplete = STAGES.every((stage) => stages[stage.id] === "completed");
  const hasFailedStage = STAGES.some((stage) => stages[stage.id] === "failed");

  return {
    ...current,
    stages,
    timings,
    currentStage: getCurrentRunningStage(stages),
    isComplete,
    isFailed: current.isFailed || hasFailedStage,
  };
}
