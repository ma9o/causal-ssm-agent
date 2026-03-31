export const STAGE4_EVENT_PREFIX = "causal-ssm.stage4.";
export const STAGE4_LOCK_NODE_ID = "__lock__";
export const STAGE4_REPAIR_BARRIER_NODE_ID = "__repair_barrier__";
export const STAGE4_DONE_NODE_ID = "__done__";

export interface Stage4GraphNode {
  id: string;
  kind: string;
  label: string;
  phase: string;
}

export interface Stage4GraphEdge {
  from: string;
  to: string;
  kind: string;
}

export interface Stage4GraphPhase {
  id: string;
  label: string;
}

export interface Stage4Graph {
  nodes: Stage4GraphNode[];
  edges: Stage4GraphEdge[];
  phases: Stage4GraphPhase[];
}

export interface Stage4Cursor {
  kind: "block" | "model_spec_lock" | "repair_barrier" | "done" | "unknown";
  block_id?: string;
  scope_block_ids?: string[];
}

export interface Stage4RepairCampaign {
  scope_kind: string;
  scope_block_ids: string[];
  completed_block_ids: string[];
}

export interface Stage4Snapshot {
  cursor: Stage4Cursor;
  block_status: Record<string, string>;
  model_spec_locked: boolean;
  repair_campaign: Stage4RepairCampaign | null;
  phase: string;
}

export interface PrefectStage4EventRecord {
  event?: string | null;
  occurred?: string | null;
  payload?: Record<string, unknown>;
}

export type Stage4Event =
  | { type: "graph"; graph: Stage4Graph }
  | { type: "snapshot"; snapshot: Stage4Snapshot };

export interface Stage4ReplayState {
  graph: Stage4Graph | null;
  snapshot: Stage4Snapshot | null;
}

export const EMPTY_STAGE4_REPLAY_STATE: Stage4ReplayState = {
  graph: null,
  snapshot: null,
};

export function getStage4StateQueryKey(workspaceId: string, rootFlowRunId: string | null) {
  return ["pipeline", workspaceId, "stage4-state", rootFlowRunId ?? "__none__"] as const;
}

export function getStage4StateQueryKeyPrefix(workspaceId: string) {
  return ["pipeline", workspaceId, "stage4-state"] as const;
}

export function parseStage4Event(
  event: PrefectStage4EventRecord | null | undefined,
): Stage4Event | null {
  if (!event?.event?.startsWith(STAGE4_EVENT_PREFIX)) return null;
  const payload = event.payload;
  if (!payload) return null;

  if (payload.type === "graph" && Array.isArray(payload.nodes) && Array.isArray(payload.edges)) {
    return {
      type: "graph",
      graph: {
        nodes: payload.nodes as Stage4GraphNode[],
        edges: payload.edges as Stage4GraphEdge[],
        phases: (payload.phases as Stage4GraphPhase[]) ?? [],
      },
    };
  }

  if (payload.type === "snapshot" && payload.cursor && payload.block_status) {
    return {
      type: "snapshot",
      snapshot: {
        cursor: payload.cursor as Stage4Cursor,
        block_status: payload.block_status as Record<string, string>,
        model_spec_locked: !!payload.model_spec_locked,
        repair_campaign: (payload.repair_campaign as Stage4RepairCampaign) ?? null,
        phase: (payload.phase as string) ?? "unknown",
      },
    };
  }

  return null;
}

export function applyStage4Event(
  state: Stage4ReplayState | undefined,
  event: Stage4Event,
): Stage4ReplayState {
  const next = state ?? EMPTY_STAGE4_REPLAY_STATE;
  if (event.type === "graph") {
    return { ...next, graph: event.graph };
  }
  return { ...next, snapshot: event.snapshot };
}

export function reduceStage4Events(events: readonly PrefectStage4EventRecord[]): Stage4ReplayState {
  return events.reduce<Stage4ReplayState>((state, record) => {
    const event = parseStage4Event(record);
    return event ? applyStage4Event(state, event) : state;
  }, EMPTY_STAGE4_REPLAY_STATE);
}
