import type { StageId } from "@causal-ssm/api-types";
import { apiFetch } from "./client";

export interface ReplayResponse {
  ok: true;
  resumeFrom: StageId | null;
  rootFlowRunId: string;
}

export interface RefineApplyResponse {
  ok: true;
  updatedFields: string[];
  resumeFrom?: StageId | null;
  rootFlowRunId?: string;
  sessionPersisted?: boolean;
}

export interface AnalysisStageExecution {
  stateType: string;
  startTime: string | null;
  endTime: string | null;
}

export interface AnalysisStageRun {
  ownerRootFlowRunId: string | null;
  stageSubflowRunId: string | null;
  initialLogFlowRunIds: string[];
  execution: AnalysisStageExecution | null;
}

export type AnalysisStageRuns = Record<StageId, AnalysisStageRun>;

export interface AnalysisManifest {
  workspaceId: string;
  createdAt: string;
  question?: string;
  rootFlowRunIds: string[];
  latestRootFlowRunId: string | null;
  stages: AnalysisStageRuns;
}

export function getAnalysisManifestQueryKey(
  workspaceId: string,
  rootFlowRunId?: string | null,
) {
  return rootFlowRunId == null
    ? (["analysis", workspaceId, "manifest"] as const)
    : (["analysis", workspaceId, "manifest", rootFlowRunId] as const);
}

export async function getAnalysisManifest(
  workspaceId: string,
  rootFlowRunId?: string | null,
): Promise<AnalysisManifest> {
  const search = rootFlowRunId
    ? `?${new URLSearchParams({ rootFlowRunId }).toString()}`
    : "";
  return apiFetch<AnalysisManifest>(`/api/analysis/${workspaceId}${search}`);
}
