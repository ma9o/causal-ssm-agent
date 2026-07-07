import type { StageId } from "@nof1-causal-lab/api-types";
import { apiFetch } from "./client";

export interface AnalysisStageExecution {
  stateType: string;
  startTime: string | null;
  endTime: string | null;
}

export interface AnalysisStageRun {
  execution: AnalysisStageExecution | null;
}

export type AnalysisStageRuns = Record<StageId, AnalysisStageRun>;

export interface MachineTransition {
  transition_id: string;
  runner_id: string;
}

export interface MachineDescription {
  transitions: MachineTransition[];
}

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

/** Per-artifact freshness status computed by the episode machine. */
export interface EpisodeArtifactStatus {
  artifact_id: string;
  exists: boolean;
  stale: boolean;
  version: number | null;
  provenance: string | null;
  produced_by: string | null;
}

export type EpisodeMove =
  | { kind: "run"; artifact_id: string }
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
  artifacts: EpisodeArtifactStatus[];
  transitions: EpisodeTransitionRecord[];
  events: EpisodeEventRecord[];
}

export function getAnalysisManifestQueryKey(workspaceId: string) {
  return ["analysis", workspaceId, "manifest"] as const;
}

export async function getAnalysisManifest(workspaceId: string): Promise<AnalysisManifest> {
  return apiFetch<AnalysisManifest>(`/api/analysis/${workspaceId}`);
}

export async function getMachineDescription(): Promise<MachineDescription> {
  return apiFetch<MachineDescription>("/api/machine");
}

export async function getEpisodeProgress(
  workspaceId: string,
  after?: string | null,
): Promise<EpisodeProgressPayload> {
  const search = after ? `?${new URLSearchParams({ after }).toString()}` : "";
  return apiFetch<EpisodeProgressPayload>(`/api/analysis/${workspaceId}/progress${search}`);
}

export async function recomputeStaleStages(
  workspaceId: string,
): Promise<{ ok: true; workspaceId: string }> {
  return apiFetch<{ ok: true; workspaceId: string }>(`/api/analysis/${workspaceId}/recompute`, {
    method: "POST",
  });
}
