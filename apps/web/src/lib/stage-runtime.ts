import type { AnalysisStageExecution, AnalysisStageRun } from "@/lib/api/analysis";
import { normalizeFlowRunIds } from "./flow-run-ids";

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

export interface StageRuntimeEventRecord {
  status: StageProgressStatus;
  occurred?: string | null;
  stageSubflowRunId?: unknown;
  logFlowRunIds?: unknown;
}

export interface StageExecutionSummary {
  execution: AnalysisStageExecution;
  stageSubflowRunId: string | null;
  initialLogFlowRunIds: string[];
}

export function normalizeStageSubflowRunId(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function normalizeLogFlowRunIds(value: unknown): string[] {
  return Array.isArray(value) ? normalizeFlowRunIds(value) : [];
}

function compareOccurredAt(left: StageRuntimeEventRecord, right: StageRuntimeEventRecord): number {
  return Date.parse(left.occurred ?? "") - Date.parse(right.occurred ?? "");
}

function toExecutionStateType(status: StageProgressStatus): AnalysisStageExecution["stateType"] {
  switch (status) {
    case "running":
      return "RUNNING";
    case "completed":
      return "COMPLETED";
    case "failed":
      return "FAILED";
  }
}

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
    case "CRASHED":
    case "CANCELLED":
    case "CANCELLING":
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

function resolveLogFlowRunIds(
  stageSubflowRunId: string | null,
  explicitLogFlowRunIds: string[],
): string[] {
  if (explicitLogFlowRunIds.length > 0) {
    return explicitLogFlowRunIds;
  }
  return stageSubflowRunId ? [stageSubflowRunId] : [];
}

function resolveStageRuntimeMetadata(
  event: Pick<StageRuntimeEventRecord, "stageSubflowRunId" | "logFlowRunIds">,
) {
  const stageSubflowRunId = normalizeStageSubflowRunId(event.stageSubflowRunId);
  const explicitLogFlowRunIds = normalizeLogFlowRunIds(event.logFlowRunIds);
  return {
    stageSubflowRunId,
    logFlowRunIds: resolveLogFlowRunIds(stageSubflowRunId, explicitLogFlowRunIds),
  };
}

export function summarizeStageProgressEvents(
  events: readonly StageRuntimeEventRecord[],
): StageExecutionSummary | null {
  if (events.length === 0) {
    return null;
  }

  const orderedEvents = [...events].sort(compareOccurredAt);
  const runningEvent = orderedEvents.find((event) => event.status === "running");
  const terminalEvent = [...orderedEvents]
    .reverse()
    .find((event) => event.status === "completed" || event.status === "failed");
  const latestEvent = terminalEvent ?? orderedEvents[orderedEvents.length - 1];
  const latestStatus = latestEvent.status;

  let stageSubflowRunId: string | null = null;
  let initialLogFlowRunIds: string[] = [];
  for (const event of orderedEvents) {
    const runtime = resolveStageRuntimeMetadata(event);
    stageSubflowRunId = runtime.stageSubflowRunId ?? stageSubflowRunId;
    if (runtime.logFlowRunIds.length > 0) {
      initialLogFlowRunIds = runtime.logFlowRunIds;
    }
  }
  if (initialLogFlowRunIds.length === 0 && stageSubflowRunId) {
    initialLogFlowRunIds = [stageSubflowRunId];
  }

  const startTime =
    runningEvent?.occurred ?? terminalEvent?.occurred ?? latestEvent.occurred ?? null;

  return {
    execution: {
      stateType: toExecutionStateType(latestStatus),
      startTime,
      endTime: latestStatus === "running" ? null : (terminalEvent?.occurred ?? startTime),
    },
    stageSubflowRunId,
    initialLogFlowRunIds,
  };
}

export function patchStageRun(
  existingStageRun: AnalysisStageRun,
  ownerRootFlowRunId: string,
  event: StageRuntimeEventRecord,
): AnalysisStageRun {
  const runtime = resolveStageRuntimeMetadata(event);
  const stageSubflowRunId = runtime.stageSubflowRunId ?? existingStageRun.stageSubflowRunId ?? null;
  const initialLogFlowRunIds =
    runtime.logFlowRunIds.length > 0
      ? runtime.logFlowRunIds
      : existingStageRun.initialLogFlowRunIds.length > 0
        ? existingStageRun.initialLogFlowRunIds
        : stageSubflowRunId
          ? [stageSubflowRunId]
          : [];

  const existingExecution = existingStageRun.execution;
  let execution = existingExecution;

  if (
    !(
      event.status === "running" &&
      existingExecution?.stateType &&
      existingExecution.stateType !== "RUNNING"
    )
  ) {
    const startTime = existingExecution?.startTime ?? event.occurred ?? null;
    execution = {
      stateType: toExecutionStateType(event.status),
      startTime,
      endTime:
        event.status === "running"
          ? null
          : (event.occurred ?? existingExecution?.endTime ?? startTime),
    };
  }

  return {
    ...existingStageRun,
    ownerRootFlowRunId,
    stageSubflowRunId,
    initialLogFlowRunIds,
    execution,
  };
}
