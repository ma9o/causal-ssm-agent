import type { StageId } from "@nof1-causal-lab/api-types";
import { apiFetch } from "./client";
import type { RefinementUIMessage } from "../utils/trace-to-core";

export interface ReplayResponse {
  ok: true;
  workspaceId: string;
}

export interface RefineApplyResponse {
  ok: true;
  updatedFields: string[];
  workspaceId?: string;
}

export interface ReplayStageOverrideRequest {
  workspaceId: string;
  stageId: StageId;
  stageData: Record<string, unknown>;
}

export interface ApplyRefinementRequest {
  workspaceId: string;
  stageId: StageId;
  stagePatch?: Record<string, unknown>;
  messages?: RefinementUIMessage[];
}

export interface AnalysisStageExecution {
  stateType: string;
  startTime: string | null;
  endTime: string | null;
}

export interface AnalysisStageRun {
  execution: AnalysisStageExecution | null;
}

export type AnalysisStageRuns = Record<StageId, AnalysisStageRun>;

export interface AnalysisManifest {
  workspaceId: string;
  createdAt: string;
  question?: string;
  stages: AnalysisStageRuns;
  /** Read-only artifact (e.g. a shared workspace): the UI hides LLM interaction. */
  readOnly: boolean;
}

/** One intra-stage telemetry event from the episode event stream. */
export interface EpisodeEventRecord {
  event: string;
  payload: Record<string, unknown>;
  cursor: string;
}

export type EpisodeMove =
  | { kind: "run"; stage_id: string }
  | { kind: "write"; artifact_id: string; provenance: string };

/** One journaled transition attempt (applied, rejected, or raised). */
export interface EpisodeTransitionRecord {
  seq: number;
  ts: string;
  move: EpisodeMove;
  status: "applied" | "rejected" | "raised";
  reason: string | null;
  error_type: string | null;
  error_message: string | null;
}

export interface EpisodeProgressPayload {
  workspaceId: string;
  autoRunning: boolean;
  seq: number;
  transitions: EpisodeTransitionRecord[];
  events: EpisodeEventRecord[];
}

export function getAnalysisManifestQueryKey(workspaceId: string) {
  return ["analysis", workspaceId, "manifest"] as const;
}

export async function getAnalysisManifest(workspaceId: string): Promise<AnalysisManifest> {
  return apiFetch<AnalysisManifest>(`/api/analysis/${workspaceId}`);
}

export async function getEpisodeProgress(
  workspaceId: string,
  after?: string | null,
): Promise<EpisodeProgressPayload> {
  const search = after ? `?${new URLSearchParams({ after }).toString()}` : "";
  return apiFetch<EpisodeProgressPayload>(`/api/analysis/${workspaceId}/progress${search}`);
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
