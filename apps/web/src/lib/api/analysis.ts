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
}

export interface RefineApplyResponse extends ReplayResponse {
  updatedFields: string[];
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
  wrapperTaskRun: AnalysisStageTaskRun | null;
}

export type AnalysisStageRuns = Record<StageId, AnalysisStageRun>;

export interface AnalysisManifest extends SessionResponse {
  userId: string;
  latestRootFlowRunId: string | null;
  stages: AnalysisStageRuns;
}

export async function getSession(userId: string): Promise<SessionResponse> {
  return apiFetch<SessionResponse>(`/api/sessions/${userId}`);
}

export function getAnalysisManifestQueryKey(userId: string) {
  return ["analysis", userId, "manifest"] as const;
}

export async function getAnalysisManifest(userId: string): Promise<AnalysisManifest> {
  return apiFetch<AnalysisManifest>(`/api/analysis/${userId}`);
}
