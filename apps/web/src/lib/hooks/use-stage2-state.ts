"use client";

import {
  EMPTY_STAGE2_REPLAY_STATE,
  getStage2RequestsPerMinute,
  getStage2StateQueryKey,
  listStage2Workers,
  summarizeStage2State,
  type Stage2ReplayState,
} from "@/lib/stage2-runtime";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

/**
 * Stage 2 worker fan-out state, reduced from polled episode telemetry
 * events into the React Query cache by use-run-events.
 */
export function useStage2State(workspaceId: string) {
  const { data } = useQuery<Stage2ReplayState>({
    queryKey: getStage2StateQueryKey(workspaceId),
    queryFn: () => undefined as never,
    enabled: false,
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
