import { type StageId, STAGES, type StageLogScopePolicy } from "@nof1-causal-lab/api-types";
import type { AnalysisStageExecution, AnalysisStageRun } from "./api/analysis";
import { buildFlowRunIdsSignature, normalizeFlowRunIds } from "./flow-run-ids";
import type { PrefectLogTimeWindow } from "./prefect-log-client";

export interface StageRuntimeRef {
  ownerRootFlowRunId: string | null;
  stageSubflowRunId: string | null;
  execution: AnalysisStageExecution | null;
}

export interface StageLogScopeDescriptor {
  runtime: StageRuntimeRef;
  initialFlowRunIds: string[];
  timeWindow: PrefectLogTimeWindow;
  refresh:
    | false
    | {
        path: string;
        intervalMs: number;
      };
}

const DYNAMIC_STAGE_LOG_SCOPE_REFRESH_MS = 3000;
const STAGE_LOG_END_PADDING_MS = 1;

export function getStageLogScopePolicy(stageId: StageId): StageLogScopePolicy {
  return STAGES.find((stage) => stage.id === stageId)?.logScopePolicy ?? "subflow";
}

export function toStageRuntimeRef(stageRun: AnalysisStageRun | null | undefined): StageRuntimeRef {
  return {
    ownerRootFlowRunId: stageRun?.ownerRootFlowRunId ?? null,
    stageSubflowRunId: stageRun?.stageSubflowRunId ?? null,
    execution: stageRun?.execution ?? null,
  };
}

export function getStageLogQueryScopeKey(runtime: StageRuntimeRef): string {
  return runtime.stageSubflowRunId ?? runtime.ownerRootFlowRunId ?? "unscoped";
}

export function getStageRuntimeInitialLogFlowRunIds(
  stageRun: AnalysisStageRun | null | undefined,
): string[] {
  const explicit = normalizeFlowRunIds(stageRun?.initialLogFlowRunIds);
  if (explicit.length > 0) {
    return explicit;
  }

  const stageSubflowRunId = stageRun?.stageSubflowRunId?.trim();
  if (stageSubflowRunId) {
    return [stageSubflowRunId];
  }

  const ownerRootFlowRunId = stageRun?.ownerRootFlowRunId?.trim();
  return ownerRootFlowRunId ? [ownerRootFlowRunId] : [];
}

export function getStageLogTimeWindow(
  execution: AnalysisStageExecution | null | undefined,
): PrefectLogTimeWindow {
  const after = execution?.startTime?.trim() || undefined;
  const endTime = execution?.endTime?.trim() || undefined;
  if (!endTime) {
    return after ? { after } : {};
  }

  const endMs = Date.parse(endTime);
  if (!Number.isFinite(endMs)) {
    return after ? { after } : {};
  }

  return {
    ...(after ? { after } : {}),
    before: new Date(endMs + STAGE_LOG_END_PADDING_MS).toISOString(),
  };
}

export function buildStageLogScopePath(
  workspaceId: string,
  stageId: StageId,
  stageSubflowRunId: string,
): string {
  const search = new URLSearchParams({ stageSubflowRunId });
  return `/api/analysis/${workspaceId}/stages/${stageId}/log-scope?${search.toString()}`;
}

export function shouldRefreshStageLogScope(
  stageId: StageId,
  isRunning: boolean,
  stageSubflowRunId: string | null,
): boolean {
  return (
    isRunning && !!stageSubflowRunId && getStageLogScopePolicy(stageId) === "subflow-with-children"
  );
}

export function getStageLogScopeRefreshIntervalMs(
  stageId: StageId,
  isRunning: boolean,
  stageSubflowRunId: string | null,
): number | false {
  return shouldRefreshStageLogScope(stageId, isRunning, stageSubflowRunId)
    ? DYNAMIC_STAGE_LOG_SCOPE_REFRESH_MS
    : false;
}

export function buildPrefectSubscriptionKey(flowRunIds: readonly string[]): string {
  return buildFlowRunIdsSignature(flowRunIds);
}

export function buildStageLogSubscriptionKey(
  flowRunIds: readonly string[],
  timeWindow: PrefectLogTimeWindow,
): string {
  return [
    buildPrefectSubscriptionKey(flowRunIds),
    timeWindow.after?.trim() ?? "",
    timeWindow.before?.trim() ?? "",
  ].join("::");
}

export function buildStageLogScopeDescriptor(
  workspaceId: string,
  stageId: StageId,
  stageRun: AnalysisStageRun | null | undefined,
  status: "pending" | "running" | "completed" | "failed",
): StageLogScopeDescriptor {
  const runtime = toStageRuntimeRef(stageRun);
  const initialFlowRunIds = getStageRuntimeInitialLogFlowRunIds(stageRun);
  const refreshIntervalMs = getStageLogScopeRefreshIntervalMs(
    stageId,
    status === "running",
    runtime.stageSubflowRunId,
  );

  return {
    runtime,
    initialFlowRunIds,
    timeWindow: getStageLogTimeWindow(runtime.execution),
    refresh:
      refreshIntervalMs !== false && runtime.stageSubflowRunId
        ? {
            path: buildStageLogScopePath(workspaceId, stageId, runtime.stageSubflowRunId),
            intervalMs: refreshIntervalMs,
          }
        : false,
  };
}
