import type { StageId } from "@causal-ssm/api-types";
import { apiFetch } from "./client";

export interface SessionResponse {
  createdAt: string;
  rootFlowRunIds: string[];
  question?: string;
}

export interface ReplayResponse {
  ok: true;
  resumeFrom: StageId | null;
  rootFlowRunId: string;
  sessionPersisted: boolean;
}

export interface RefineApplyResponse extends ReplayResponse {
  updatedFields: string[];
}

export interface WorkspaceUnlockResponse {
  ok: true;
}

export interface AnalysisStageTaskRun {
  id: string;
  name: string;
  stateType: string;
  startTime: string | null;
  endTime: string | null;
}

export interface AnalysisStageRun {
  ownerRootFlowRunId: string | null;
  stageSubflowRunId: string | null;
  logFlowRunIds: string[];
  wrapperTaskRun: AnalysisStageTaskRun | null;
}

export type AnalysisStageRuns = Record<StageId, AnalysisStageRun>;

export interface AnalysisManifest extends SessionResponse {
  workspaceId: string;
  latestRootFlowRunId: string | null;
  stages: AnalysisStageRuns;
}

export async function getSession(workspaceId: string): Promise<SessionResponse> {
  return apiFetch<SessionResponse>(`/api/sessions/${workspaceId}`);
}

export async function unlockWorkspace(
  workspaceId: string,
  accessCode: string,
): Promise<WorkspaceUnlockResponse> {
  return apiFetch<WorkspaceUnlockResponse>("/api/workspaces/unlock", {
    method: "POST",
    body: JSON.stringify({ workspaceId, accessCode }),
  });
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
