import { getToolServerUrl } from "@/lib/runtime-urls";
import { createByokSecretRef } from "@/lib/server/byok-secret-store";
import { noAccessMessage, resolveOpenRouterAccess } from "@/lib/server/openrouter-access";

const TOOL_SERVER = getToolServerUrl();

export type EpisodeProvenance = "computed" | "human" | "llm";

export type EpisodeArtifactId =
  | "question"
  | "raw_data"
  | "constructs"
  | "causal_spec"
  | "identification_report"
  | "estimands"
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

/** Per-move execution parameters accepted by the facade (infra, not domain state). */
export interface EpisodeExecOptions {
  openrouter_access_mode?: string;
  openrouter_secret_ref?: string;
}

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
  "stage-1a": "constructs",
  "stage-1b": "causal_spec",
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
  options?: EpisodeExecOptions,
): Promise<MoveOutcome> {
  return episodeFetch(`/${workspaceId}/moves`, {
    method: "POST",
    body: JSON.stringify({
      move,
      ...(payload !== undefined ? { payload } : {}),
      ...(options !== undefined ? { options } : {}),
    }),
  });
}

/** Starts the background auto-run driver; throws EpisodeRunError(409) when already running. */
export async function startAutoRun(
  workspaceId: string,
  options: EpisodeExecOptions,
): Promise<void> {
  await episodeFetch(`/${workspaceId}/auto`, {
    method: "POST",
    body: JSON.stringify({ options }),
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
 * Resolve OpenRouter access into the exec options for an auto-run.
 *
 * Local mode uses ambient credentials on the worker. For user/anonymous modes
 * the key is handed off as a single-use encrypted secret ref.
 * TODO: a secret ref authorizes exactly one move, so an auto-run spanning
 * multiple LLM stages exhausts it after the first stage; per-move secret refs
 * are not supported by the facade yet.
 */
export async function resolveAutoRunExecOptions(): Promise<EpisodeExecOptions> {
  const access = await resolveOpenRouterAccess();
  if (access.mode === "none") {
    throw new EpisodeRunError(402, noAccessMessage(access.reason));
  }
  if (access.mode === "local") {
    return { openrouter_access_mode: "local" };
  }

  let secretRef: string;
  try {
    secretRef = await createByokSecretRef(access.apiKey);
  } catch {
    throw new EpisodeRunError(500, "OpenRouter secret handoff is not configured.");
  }
  return {
    openrouter_access_mode: access.mode,
    openrouter_secret_ref: secretRef,
  };
}
