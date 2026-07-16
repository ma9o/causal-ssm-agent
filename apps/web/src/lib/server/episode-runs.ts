import type { CapabilitiesResponse, LLMTrace } from "@nof1-causal-lab/api-types";
import type {
  EpisodeArtifactId,
  EpisodeEvent,
  EpisodeMove,
  EpisodeStatus,
  JsonObject,
  MachineDescription,
  MoveOutcome,
  TransitionRecord,
} from "@/lib/episode-types";
import { getToolServerUrl } from "@/lib/runtime-urls";

export type {
  EpisodeArtifactId,
  EpisodeArtifactStatus,
  EpisodeEvent,
  EpisodeMove,
  EpisodeProvenance,
  EpisodeState,
  EpisodeStatus,
  JsonObject,
  MachineDescription,
  MoveOutcome,
  RetractedArtifact,
  TransitionRecord,
  TransitionStatus,
} from "@/lib/episode-types";

const TOOL_SERVER = getToolServerUrl();

/** Artifacts a human-edited result can write back into the machine. */
export const WRITABLE_ARTIFACTS: Partial<Record<string, EpisodeArtifactId>> = {
  latent_structure: "latent_structure",
  measurement_structure: "measurement_structure",
  statistical_model_spec: "statistical_model_spec",
  baseline_report: "baseline_report",
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
  payload?: JsonObject,
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

export interface TransitionTraceIndex {
  workspace_id: string;
  artifact_id: string;
  version: number;
  seq: number;
  trace_ids: string[];
}

export async function getArtifactTraceIndex(
  workspaceId: string,
  artifactId: string,
): Promise<TransitionTraceIndex> {
  const response = await fetch(
    `${TOOL_SERVER}/api/episodes/${workspaceId}/artifacts/${artifactId}/traces`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw new EpisodeRunError(
      response.status === 404 ? 404 : 502,
      `Trace API error ${response.status}: ${await response.text()}`,
    );
  }
  return response.json() as Promise<TransitionTraceIndex>;
}

export async function getEpisodeTrace(
  workspaceId: string,
  seq: number,
  subroutineId: string,
): Promise<LLMTrace> {
  const response = await fetch(
    `${TOOL_SERVER}/api/episodes/${workspaceId}/traces/${seq}/${encodeURIComponent(subroutineId)}`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw new EpisodeRunError(
      response.status === 404 ? 404 : 502,
      `Trace API error ${response.status}: ${await response.text()}`,
    );
  }
  return response.json() as Promise<LLMTrace>;
}

export async function getMachineDescription(): Promise<MachineDescription> {
  const response = await fetch(`${TOOL_SERVER}/api/machine`, { cache: "no-store" });
  if (!response.ok) {
    throw new EpisodeRunError(502, `Machine description error ${response.status}`);
  }
  return response.json() as Promise<MachineDescription>;
}

/**
 * Facade deployment capabilities. A read-only facade (the hosted viewer's
 * backend) reports moves_enabled=false; the UI hides move affordances.
 */
export async function getFacadeCapabilities(): Promise<CapabilitiesResponse> {
  const response = await fetch(`${TOOL_SERVER}/api/capabilities`, { cache: "no-store" });
  if (!response.ok) {
    throw new EpisodeRunError(502, `Capabilities error ${response.status}`);
  }
  return response.json() as Promise<CapabilitiesResponse>;
}
