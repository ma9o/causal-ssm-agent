"use client";

import { getStage4ReplayState } from "@/lib/api/analysis";
import {
  EMPTY_STAGE4_REPLAY_STATE,
  getStage4StateQueryKey,
  type Stage4Graph,
  type Stage4Snapshot,
} from "@/lib/stage4-runtime";
import { useQuery } from "@tanstack/react-query";
import type { StageRunStatus } from "./use-run-events";

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
 * Stage 4 state-machine graph and live snapshot via WebSocket events.
 *
 * The static graph topology and live snapshot are populated by Prefect custom
 * events in use-run-events.ts and written into the React Query cache.
 */
export function useStage4Graph(
  workspaceId: string,
  stageStatus: StageRunStatus,
  rootFlowRunId: string | null,
) {
  const isActive = stageStatus === "running";

  const { data } = useQuery({
    queryKey: getStage4StateQueryKey(workspaceId, rootFlowRunId),
    queryFn: () =>
      rootFlowRunId
        ? getStage4ReplayState(workspaceId, rootFlowRunId)
        : Promise.resolve(EMPTY_STAGE4_REPLAY_STATE),
    enabled: isActive && !!rootFlowRunId,
    staleTime: Infinity,
  });

  return {
    graph: data?.graph && data.graph.nodes.length > 0 ? data.graph : null,
    snapshot: data?.snapshot ?? null,
    lastBlockStateById: data?.lastBlockStateById ?? {},
  };
}
