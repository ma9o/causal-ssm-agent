import type { StageId } from "@nof1-causal-lab/api-types";
import type { Stage2ReplayState } from "../stage2-runtime";
import type { Stage4ReplayState } from "../stage4-runtime";
import { apiFetch } from "./client";
import type { RefinementUIMessage } from "../utils/trace-to-core";

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

export interface ReplayStageOverrideRequest {
  workspaceId: string;
  stageId: StageId;
  stageData: Record<string, unknown>;
  rootFlowRunId?: string | null;
}

export interface ApplyRefinementRequest {
  workspaceId: string;
  stageId: StageId;
  stagePatch?: Record<string, unknown>;
  messages?: RefinementUIMessage[];
  rootFlowRunId?: string | null;
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
  /** Read-only artifact (e.g. a shared workspace): the UI hides LLM interaction. */
  readOnly: boolean;
}

export function getAnalysisManifestQueryKey(workspaceId: string, rootFlowRunId?: string | null) {
  return rootFlowRunId == null
    ? (["analysis", workspaceId, "manifest"] as const)
    : (["analysis", workspaceId, "manifest", rootFlowRunId] as const);
}

export async function getAnalysisManifest(
  workspaceId: string,
  rootFlowRunId?: string | null,
): Promise<AnalysisManifest> {
  const search = rootFlowRunId ? `?${new URLSearchParams({ rootFlowRunId }).toString()}` : "";
  return apiFetch<AnalysisManifest>(`/api/analysis/${workspaceId}${search}`);
}

export async function getStage4ReplayState(
  workspaceId: string,
  rootFlowRunId: string,
): Promise<Stage4ReplayState> {
  const search = new URLSearchParams({ rootFlowRunId }).toString();
  return apiFetch<Stage4ReplayState>(`/api/analysis/${workspaceId}/stage4-state?${search}`);
}

export async function getStage2ReplayState(
  workspaceId: string,
  rootFlowRunId: string,
): Promise<Stage2ReplayState> {
  const search = new URLSearchParams({ rootFlowRunId }).toString();
  return apiFetch<Stage2ReplayState>(`/api/analysis/${workspaceId}/stage2-state?${search}`);
}

export async function replayStageOverride(
  payload: ReplayStageOverrideRequest,
): Promise<ReplayResponse> {
  return apiFetch<ReplayResponse>("/api/replay", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function applyRefinement(
  payload: ApplyRefinementRequest,
): Promise<RefineApplyResponse> {
  return apiFetch<RefineApplyResponse>("/api/refine/apply", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
