import { type StageId, STAGES, type StageLogScopePolicy } from "@causal-ssm/api-types";
import type { AnalysisStageExecution, AnalysisStageRun } from "./api/analysis";
import { buildFlowRunIdsSignature, normalizeFlowRunIds } from "./flow-run-ids";

export interface StageRuntimeRef {
  ownerRootFlowRunId: string | null;
  stageSubflowRunId: string | null;
  execution: AnalysisStageExecution | null;
}

const DYNAMIC_STAGE_LOG_SCOPE_REFRESH_MS = 3000;

export function getStageLogScopePolicy(stageId: StageId): StageLogScopePolicy {
  return STAGES.find((stage) => stage.id === stageId)?.logScopePolicy ?? "subflow";
}

export function toStageRuntimeRef(
  stageRun: AnalysisStageRun | null | undefined,
): StageRuntimeRef {
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
  return stageSubflowRunId ? [stageSubflowRunId] : [];
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
    isRunning &&
    !!stageSubflowRunId &&
    getStageLogScopePolicy(stageId) === "subflow-with-children"
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
