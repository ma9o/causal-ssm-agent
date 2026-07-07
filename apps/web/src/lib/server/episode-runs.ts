import { getToolServerUrl } from "@/lib/runtime-urls";

const TOOL_SERVER = getToolServerUrl();

export type EpisodeProvenance = "computed" | "human" | "llm";

export type EpisodeArtifactId =
  | "question"
  | "raw_data"
  | "latent_structure"
  | "causal_design"
  | "identification_report"
  | "extraction_report"
  | "model_data"
  | "validation_report"
  | "compiled_ssm"
  | "posterior"
  | "baseline_ranking"
  | "saved_scenarios";

export type EpisodeMove =
  | { kind: "run"; stage_id: string }
  | { kind: "write"; artifact_id: EpisodeArtifactId; provenance: EpisodeProvenance };

export interface ArtifactVersionInfo {
  artifact_id: EpisodeArtifactId;
  version: number;
  provenance: EpisodeProvenance;
  derived_from: Partial<Record<EpisodeArtifactId, number>>;
  produced_by: string | null;
  created_at: string;
}

export interface EpisodeState {
  current: Partial<Record<EpisodeArtifactId, ArtifactVersionInfo>>;
}

export type TransitionStatus = "applied" | "rejected" | "raised";

export interface TransitionRecord {
  seq: number;
  ts: string;
  move: EpisodeMove;
  status: TransitionStatus;
  reason: string | null;
  error_type: string | null;
  error_message: string | null;
  diagnostics: Record<string, unknown>;
  produced: ArtifactVersionInfo[];
  retracted: EpisodeArtifactId[];
  state_after: EpisodeState;
}

export interface MoveOutcome {
  seq: number;
  status: TransitionStatus;
  reason: string | null;
  error_type: string | null;
  error_message: string | null;
  diagnostics: Record<string, unknown>;
  produced: ArtifactVersionInfo[];
  retracted: EpisodeArtifactId[];
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
  payload: Record<string, unknown>;
  cursor: string;
}

/** The artifact a stage's human-edited result writes back into the machine. */
export const STAGE_EDIT_ARTIFACTS: Partial<Record<string, EpisodeArtifactId>> = {
  "stage-1a": "latent_structure",
  "stage-1b": "causal_design",
  "stage-6": "baseline_ranking",
};

export class EpisodeRunError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function episodeFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${TOOL_SERVER}/api/episodes${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new EpisodeRunError(
      response.status === 409 ? 409 : 502,
      `Episode API error ${response.status}: ${await response.text()}`,
    );
  }
  return response.json() as Promise<T>;
}

export async function startEpisode(
  workspaceId: string,
  question?: string,
): Promise<EpisodeStatus & { ok: boolean; outcome: MoveOutcome | null }> {
  return episodeFetch("", {
    method: "POST",
    body: JSON.stringify({
      workspace_id: workspaceId,
      ...(question !== undefined ? { question } : {}),
    }),
  });
}

export async function proposeMove(
  workspaceId: string,
  move: EpisodeMove,
  payload?: Record<string, unknown>,
): Promise<MoveOutcome> {
  return episodeFetch(`/${workspaceId}/moves`, {
    method: "POST",
    body: JSON.stringify({
      move,
      ...(payload !== undefined ? { payload } : {}),
    }),
  });
}

/** Starts the background auto-run driver; throws EpisodeRunError(409) when already running. */
export async function startAutoRun(workspaceId: string): Promise<void> {
  await episodeFetch(`/${workspaceId}/auto`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export async function getEpisodeStatus(workspaceId: string): Promise<EpisodeStatus> {
  return episodeFetch(`/${workspaceId}`);
}

export async function getEpisodeTimeline(
  workspaceId: string,
): Promise<{ workspace_id: string; transitions: TransitionRecord[] }> {
  return episodeFetch(`/${workspaceId}/timeline`);
}

export async function getEpisodeEvents(
  workspaceId: string,
  after?: string | null,
): Promise<{ workspace_id: string; events: EpisodeEvent[] }> {
  const search = after ? `?${new URLSearchParams({ after }).toString()}` : "";
  return episodeFetch(`/${workspaceId}/events${search}`);
}

/**
 * Facade deployment capabilities. A read-only facade (the hosted viewer's
 * backend) reports moves_enabled=false; the UI hides move affordances.
 */
export async function getFacadeCapabilities(): Promise<{ moves_enabled: boolean }> {
  const response = await fetch(`${TOOL_SERVER}/api/capabilities`, { cache: "no-store" });
  if (!response.ok) {
    throw new EpisodeRunError(502, `Capabilities error ${response.status}`);
  }
  return response.json() as Promise<{ moves_enabled: boolean }>;
}
