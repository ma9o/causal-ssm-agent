import type { ArtifactViewId } from "./transitions";

export type ArtifactStatus = "pending" | "running" | "completed" | "failed" | "blocked";

export type RunStatus = "pending" | "running" | "completed" | "failed";

export interface ArtifactViewState {
  id: ArtifactViewId;
  status: ArtifactStatus;
  startedAt: string | null;
  completedAt: string | null;
  error: string | null;
}

export interface PipelineRun {
  id: string;
  status: RunStatus;
  question: string;
  createdAt: string;
  artifacts: ArtifactViewState[];
}
