import type { StageId } from "./stages";

export type StageStatus = "pending" | "running" | "completed" | "failed" | "blocked";

export type RunStatus = "pending" | "running" | "completed" | "failed";

export interface StageState {
  id: StageId;
  status: StageStatus;
  startedAt: string | null;
  completedAt: string | null;
  error: string | null;
}

export interface PipelineRun {
  id: string;
  status: RunStatus;
  question: string;
  createdAt: string;
  stages: StageState[];
}
