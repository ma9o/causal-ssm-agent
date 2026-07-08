"use client";

import {
  EMPTY_EXTRACTION_REPLAY_STATE,
  getExtractionRequestsPerMinute,
  getExtractionStateQueryKey,
  listExtractionWorkers,
  summarizeExtractionState,
  type ExtractionReplayState,
} from "@/lib/extraction-runtime";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

export function useExtractionState(workspaceId: string) {
  const { data } = useQuery<ExtractionReplayState>({
    queryKey: getExtractionStateQueryKey(workspaceId),
    queryFn: () => undefined as never,
    enabled: false,
  });

  return useMemo(() => {
    const state = data ?? EMPTY_EXTRACTION_REPLAY_STATE;
    return {
      state,
      plan: state.plan,
      workers: listExtractionWorkers(state),
      summary: summarizeExtractionState(state),
      rpm: getExtractionRequestsPerMinute(state),
      maxRpm:
        typeof state.plan?.max_rpm === "number" && state.plan.max_rpm > 0
          ? state.plan.max_rpm
          : 450,
    };
  }, [data]);
}
