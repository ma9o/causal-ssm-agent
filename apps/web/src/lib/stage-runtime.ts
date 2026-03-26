import type { AnalysisStageExecution, AnalysisStageRun } from "@/lib/api/analysis";

export const STAGE_PROGRESS_EVENT_PREFIX = "causal-ssm.pipeline-stage";
export const STAGE_PROGRESS_EVENT_FILTER_PREFIX = `${STAGE_PROGRESS_EVENT_PREFIX}.`;

export type StageProgressStatus = "running" | "completed" | "failed";

export interface StageRuntimeEventRecord {
  status: StageProgressStatus;
  occurred?: string | null;
  stageSubflowRunId?: unknown;
  logFlowRunIds?: unknown;
}

export interface StageExecutionSummary {
  execution: AnalysisStageExecution;
  stageSubflowRunId: string | null;
  logFlowRunIds: string[];
}

export function normalizeStageSubflowRunId(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function normalizeLogFlowRunIds(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const ids: string[] = [];
  const seen = new Set<string>();
  for (const item of value) {
    if (typeof item !== "string") {
      continue;
    }
    const normalized = item.trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    ids.push(normalized);
  }

  return ids;
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

function resolveLogFlowRunIds(
  stageSubflowRunId: string | null,
  explicitLogFlowRunIds: string[],
): string[] {
  if (explicitLogFlowRunIds.length > 0) {
    return explicitLogFlowRunIds;
  }
  return stageSubflowRunId ? [stageSubflowRunId] : [];
}

function resolveStageRuntimeMetadata(event: Pick<StageRuntimeEventRecord, "stageSubflowRunId" | "logFlowRunIds">) {
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
  let logFlowRunIds: string[] = [];
  for (const event of orderedEvents) {
    const runtime = resolveStageRuntimeMetadata(event);
    stageSubflowRunId = runtime.stageSubflowRunId ?? stageSubflowRunId;
    if (runtime.logFlowRunIds.length > 0) {
      logFlowRunIds = runtime.logFlowRunIds;
    }
  }
  if (logFlowRunIds.length === 0 && stageSubflowRunId) {
    logFlowRunIds = [stageSubflowRunId];
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
    logFlowRunIds,
  };
}

export function patchStageRun(
  existingStageRun: AnalysisStageRun,
  ownerRootFlowRunId: string,
  event: StageRuntimeEventRecord,
): AnalysisStageRun {
  const runtime = resolveStageRuntimeMetadata(event);
  const stageSubflowRunId = runtime.stageSubflowRunId ?? existingStageRun.stageSubflowRunId ?? null;
  const logFlowRunIds =
    runtime.logFlowRunIds.length > 0
      ? runtime.logFlowRunIds
      : existingStageRun.logFlowRunIds.length > 0
        ? existingStageRun.logFlowRunIds
        : stageSubflowRunId
          ? [stageSubflowRunId]
          : [];

  const existingExecution = existingStageRun.execution;
  let execution = existingExecution;

  if (!(event.status === "running" && existingExecution?.stateType && existingExecution.stateType !== "RUNNING")) {
    const startTime = existingExecution?.startTime ?? event.occurred ?? null;
    execution = {
      stateType: toExecutionStateType(event.status),
      startTime,
      endTime: event.status === "running" ? null : (event.occurred ?? existingExecution?.endTime ?? startTime),
    };
  }

  return {
    ...existingStageRun,
    ownerRootFlowRunId,
    stageSubflowRunId,
    logFlowRunIds,
    execution,
  };
}
