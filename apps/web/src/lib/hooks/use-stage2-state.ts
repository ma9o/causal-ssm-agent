"use client";

import { getStage2ReplayState } from "@/lib/api/analysis";
import {
  EMPTY_STAGE2_REPLAY_STATE,
  getStage2RequestsPerMinute,
  getStage2StateQueryKey,
  listStage2Workers,
  summarizeStage2State,
} from "@/lib/stage2-runtime";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import type { StageRunStatus } from "./use-run-events";

export function useStage2State(
  workspaceId: string,
  stageStatus: StageRunStatus,
  rootFlowRunId: string | null,
) {
  const isActive = stageStatus === "running";

  const { data } = useQuery({
    queryKey: getStage2StateQueryKey(workspaceId, rootFlowRunId),
    queryFn: () =>
      rootFlowRunId
        ? getStage2ReplayState(workspaceId, rootFlowRunId)
        : Promise.resolve(EMPTY_STAGE2_REPLAY_STATE),
    enabled: isActive && !!rootFlowRunId,
    staleTime: Infinity,
  });

  return useMemo(() => {
    const state = data ?? EMPTY_STAGE2_REPLAY_STATE;
    return {
      state,
      plan: state.plan,
      workers: listStage2Workers(state),
      summary: summarizeStage2State(state),
      rpm: getStage2RequestsPerMinute(state),
      maxRpm:
        typeof state.plan?.max_rpm === "number" && state.plan.max_rpm > 0
          ? state.plan.max_rpm
          : 450,
    };
  }, [data]);
}
