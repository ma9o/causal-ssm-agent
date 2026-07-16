import type { ArtifactId, ArtifactVersionInfo } from "@nof1-causal-lab/api-types";

export type EpisodeArtifactId = ArtifactId;
export type EpisodeProvenance = "computed" | "human" | "llm";
export type JsonScalar = null | boolean | number | string;
export type JsonValue = JsonScalar | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export type EpisodeMove =
  | { kind: "run"; artifact_id: EpisodeArtifactId }
  | { kind: "write"; artifact_id: EpisodeArtifactId; provenance: EpisodeProvenance };

export interface EpisodeState {
  current: Partial<Record<EpisodeArtifactId, ArtifactVersionInfo>>;
}

export type TransitionStatus = "applied" | "rejected" | "raised";

export interface RetractedArtifact {
  artifact_id: EpisodeArtifactId;
  reason_ref: string;
}

export interface ResumeRef {
  kind: "model_spec";
  run_id: string;
  checkpoint_id: string;
}

export interface TransitionRecord {
  seq: number;
  ts: string;
  move: EpisodeMove;
  status: TransitionStatus;
  reason: string | null;
  error_type: string | null;
  error_message: string | null;
  diagnostics: JsonObject;
  produced: ArtifactVersionInfo[];
  retracted: RetractedArtifact[];
  trace_ids: string[];
  resume: ResumeRef | null;
}

export type EpisodeTransitionRecord = Pick<
  TransitionRecord,
  "seq" | "ts" | "move" | "status" | "reason" | "error_type" | "error_message"
>;

export interface MoveOutcome
  extends Omit<TransitionRecord, "ts" | "move" | "trace_ids" | "resume"> {
  state: EpisodeState;
}

export interface EpisodeArtifactStatus {
  artifact_id: EpisodeArtifactId;
  exists: boolean;
  stale: boolean;
  version: number | null;
  provenance: EpisodeProvenance | null;
  produced_by: string | null;
}

export interface EpisodeStatus {
  workspace_id: string;
  seq: number;
  state: EpisodeState;
  artifacts: EpisodeArtifactStatus[];
  legal: EpisodeMove[];
  auto_running: boolean;
}

export interface EpisodeEvent {
  event: string;
  payload: JsonObject;
  cursor: string;
}

export interface MachineTransition {
  transition_id: EpisodeArtifactId;
}

export interface MachineDescription {
  topological_artifact_order: EpisodeArtifactId[];
  topological_transition_order: EpisodeArtifactId[];
  transitions: MachineTransition[];
}
