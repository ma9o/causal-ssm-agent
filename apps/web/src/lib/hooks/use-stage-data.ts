"use client";

import type { StageId } from "@causal-ssm/api-types";
import { useQuery } from "@tanstack/react-query";
import { getStageResult } from "../api/endpoints";
import { isMockMode } from "../api/mock-provider";

async function fetchStageData<T>(runId: string, stage: StageId): Promise<T> {
  let payload: unknown;

  if (isMockMode()) {
    const res = await fetch(`/api/results/${runId}/${stage}`);
    if (!res.ok) throw new Error(`Mock data not found for ${stage}`);
    payload = await res.json();
  } else {
    payload = await getStageResult<unknown>(runId, stage);
  }

  return payload as T;
}

export function useStageData<T>(runId: string | null, stage: StageId, enabled: boolean) {
  return useQuery<T>({
    queryKey: ["pipeline", runId, "stage", stage],
    queryFn: () => fetchStageData<T>(runId as string, stage),
    enabled: !!runId && enabled,
    staleTime: Number.POSITIVE_INFINITY,
  });
}
