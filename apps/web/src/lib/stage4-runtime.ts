export const STAGE4_EVENT_PREFIX = "nof1-causal-lab.stage4.";
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

export interface Stage4TransitionPrior {
  parameter: string;
  distribution?: string;
  params?: Record<string, unknown>;
  reasoning?: string;
}

export interface Stage4BlockLastState {
  block_id: string;
  status: "accepted" | "reopened";
  detail_kind: "indicator_choice" | "prior_bundle" | "review_approval" | "revision";
  variable?: string;
  distribution?: string;
  link?: string;
  reasoning?: string;
  parameter_names?: string[];
  priors?: Stage4TransitionPrior[];
  reason?: string;
  scope_kind?: string;
}

export interface PrefectStage4EventRecord {
  event?: string | null;
  occurred?: string | null;
  payload?: Record<string, unknown>;
}

export type Stage4Event =
  | { type: "graph"; graph: Stage4Graph }
  | { type: "snapshot"; snapshot: Stage4Snapshot }
  | { type: "block_transition"; transition: Stage4BlockLastState };

export interface Stage4ReplayState {
  graph: Stage4Graph | null;
  snapshot: Stage4Snapshot | null;
  lastBlockStateById: Record<string, Stage4BlockLastState>;
}

export const EMPTY_STAGE4_REPLAY_STATE: Stage4ReplayState = {
  graph: null,
  snapshot: null,
  lastBlockStateById: {},
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
        phases: (payload.phases ?? []) as Stage4GraphPhase[],
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
        repair_campaign: (payload.repair_campaign ?? null) as Stage4RepairCampaign | null,
        phase: (payload.phase as string) ?? "unknown",
      },
    };
  }

  if (
    payload.type === "block_transition" &&
    typeof payload.block_id === "string" &&
    (payload.status === "accepted" || payload.status === "reopened")
  ) {
    return {
      type: "block_transition",
      transition: {
        block_id: payload.block_id,
        status: payload.status,
        detail_kind: (payload.detail_kind as Stage4BlockLastState["detail_kind"]) ?? "revision",
        variable: typeof payload.variable === "string" ? payload.variable : undefined,
        distribution: typeof payload.distribution === "string" ? payload.distribution : undefined,
        link: typeof payload.link === "string" ? payload.link : undefined,
        reasoning: typeof payload.reasoning === "string" ? payload.reasoning : undefined,
        parameter_names: Array.isArray(payload.parameter_names)
          ? payload.parameter_names.filter((value): value is string => typeof value === "string")
          : undefined,
        priors: Array.isArray(payload.priors)
          ? payload.priors
              .filter(
                (value): value is Record<string, unknown> => !!value && typeof value === "object",
              )
              .map((prior) => ({
                parameter: String(prior.parameter ?? ""),
                distribution:
                  typeof prior.distribution === "string" ? prior.distribution : undefined,
                params:
                  prior.params && typeof prior.params === "object"
                    ? (prior.params as Record<string, unknown>)
                    : undefined,
                reasoning: typeof prior.reasoning === "string" ? prior.reasoning : undefined,
              }))
              .filter((prior) => prior.parameter.length > 0)
          : undefined,
        reason: typeof payload.reason === "string" ? payload.reason : undefined,
        scope_kind: typeof payload.scope_kind === "string" ? payload.scope_kind : undefined,
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
  if (event.type === "snapshot") {
    return { ...next, snapshot: event.snapshot };
  }
  return {
    ...next,
    lastBlockStateById: {
      ...next.lastBlockStateById,
      [event.transition.block_id]: event.transition,
    },
  };
}

export function reduceStage4Events(events: readonly PrefectStage4EventRecord[]): Stage4ReplayState {
  return events.reduce<Stage4ReplayState>((state, record) => {
    const event = parseStage4Event(record);
    return event ? applyStage4Event(state, event) : state;
  }, EMPTY_STAGE4_REPLAY_STATE);
}
