"use client";

import {
  getStage4StateQueryKey,
  type Stage4Graph,
  type Stage4ReplayState,
  type Stage4Snapshot,
} from "@/lib/stage4-runtime";
import { useQuery } from "@tanstack/react-query";

export type { Stage4Graph, Stage4Snapshot };
export type {
  Stage4BlockLastState,
  Stage4GraphNode,
  Stage4GraphEdge,
  Stage4GraphPhase,
  Stage4Cursor,
  Stage4RepairCampaign,
} from "@/lib/stage4-runtime";
export {
  STAGE4_LOCK_NODE_ID,
  STAGE4_REPAIR_BARRIER_NODE_ID,
  STAGE4_DONE_NODE_ID,
} from "@/lib/stage4-runtime";

/**
 * Stage 4 state-machine graph and live snapshot, reduced from polled
 * episode telemetry events into the React Query cache by use-run-events.
 */
export function useStage4Graph(workspaceId: string) {
  const { data } = useQuery<Stage4ReplayState>({
    queryKey: getStage4StateQueryKey(workspaceId),
    queryFn: () => undefined as never,
    enabled: false,
  });

  return {
    graph: data?.graph && data.graph.nodes.length > 0 ? data.graph : null,
    snapshot: data?.snapshot ?? null,
    lastBlockStateById: data?.lastBlockStateById ?? {},
  };
}
