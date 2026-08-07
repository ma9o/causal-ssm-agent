import type { ArtifactViewId } from "@nof1-causal-lab/api-types";
import type {
  EpisodeArtifactStatus,
  EpisodeEvent as EpisodeEventRecord,
  EpisodeTransitionRecord,
} from "@/lib/episode-types";
export type {
  EpisodeArtifactStatus,
  EpisodeEvent as EpisodeEventRecord,
  EpisodeMove,
  EpisodeTransitionRecord,
} from "@/lib/episode-types";
import { apiFetch } from "./client";

export interface AnalysisTransitionExecution {
  stateType: string;
  startTime: string | null;
  endTime: string | null;
}

export interface AnalysisTransitionRun {
  execution: AnalysisTransitionExecution | null;
}

export type AnalysisTransitionRuns = Record<ArtifactViewId, AnalysisTransitionRun>;

export interface AnalysisManifest {
  workspaceId: string;
  createdAt: string;
  question?: string;
  transitionOrder: ArtifactViewId[];
  transitionRuns: AnalysisTransitionRuns;
  /** Read-only artifact (e.g. a shared workspace): the UI hides LLM interaction. */
  readOnly: boolean;
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

export async function getEpisodeProgress(
  workspaceId: string,
  after?: string | null,
): Promise<EpisodeProgressPayload> {
  const search = after ? `?${new URLSearchParams({ after }).toString()}` : "";
  return apiFetch<EpisodeProgressPayload>(`/api/analysis/${workspaceId}/progress${search}`);
}

export async function recomputeStaleArtifacts(
  workspaceId: string,
): Promise<{ ok: true; workspaceId: string }> {
  return apiFetch<{ ok: true; workspaceId: string }>(`/api/analysis/${workspaceId}/recompute`, {
    method: "POST",
  });
}
