import type { StageId, StageStatus } from "@nof1-causal-lab/api-types";
import { STAGES } from "@nof1-causal-lab/api-types";

export type StageRunStatus = Exclude<StageStatus, "blocked">;

export interface StageTiming {
  startedAt: number;
  completedAt?: number;
}

export interface PipelineProgress {
  stages: Record<StageId, StageRunStatus>;
  timings: Partial<Record<StageId, StageTiming>>;
  /** Failure detail per stage (raised transition / failed telemetry event). */
  stageErrors: Partial<Record<StageId, string>>;
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
    stageErrors: {},
    currentStage: null,
    isComplete: false,
    isFailed: false,
  };
}

/**
 * A `running` telemetry event begins a new attempt: unlike applyStageUpdate
 * (which merges unordered signals by priority), the ordered event stream may
 * legitimately re-run a completed or failed stage after its inputs changed,
 * so a terminal state is reset rather than preserved.
 */
export function restartStageAttempt(
  prev: PipelineProgress | undefined,
  stageId: StageId,
  eventTime?: number,
): PipelineProgress {
  const current = prev ?? initialProgress();
  if (current.stages[stageId] === "running") {
    return applyStageUpdate(current, stageId, "running", eventTime);
  }

  const ts = eventTime ?? Date.now();
  const stages = { ...current.stages, [stageId]: "running" as StageRunStatus };
  const stageErrors = { ...current.stageErrors };
  delete stageErrors[stageId];

  return {
    ...current,
    stages,
    timings: { ...current.timings, [stageId]: { startedAt: ts } },
    stageErrors,
    currentStage: getCurrentRunningStage(stages),
    isComplete: false,
    isFailed: STAGES.some((stage) => stages[stage.id] === "failed"),
  };
}

export function mapExecutionStateType(stateType: string): StageRunStatus | null {
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

export function applyStageUpdate(
  prev: PipelineProgress | undefined,
  stageId: StageId,
  status: StageRunStatus,
  eventTime?: number,
  errorMessage?: string,
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

  const stageErrors =
    status === "failed" && errorMessage
      ? { ...current.stageErrors, [stageId]: errorMessage }
      : current.stageErrors;

  return {
    ...current,
    stages,
    timings,
    stageErrors,
    currentStage: getCurrentRunningStage(stages),
    isComplete,
    isFailed: current.isFailed || hasFailedStage,
  };
}
