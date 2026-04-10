"use client";

import type { StageId } from "@causal-ssm/api-types";
import { useQuery } from "@tanstack/react-query";
import { getStageResult } from "../api/endpoints";
import { isMockMode } from "../api/mock-provider";

const STAGE_DATA_QUERY_VERSION = 2;

async function fetchStageData<T>(workspaceId: string, stage: StageId): Promise<T> {
  let payload: unknown;

  if (isMockMode()) {
    const res = await fetch(`/api/results/${workspaceId}/${stage}`);
    if (!res.ok) throw new Error(`Mock data not found for ${stage}`);
    payload = await res.json();
  } else {
    payload = await getStageResult<unknown>(workspaceId, stage);
  }

  return payload as T;
}

export function getStageDataQueryKey(workspaceId: string | null, stage: StageId) {
  return ["pipeline", workspaceId, "stage", stage, `v${STAGE_DATA_QUERY_VERSION}`] as const;
}

/**
 * Fetch stage data once after completion and cache indefinitely.
 */
export function useStageData<T>(workspaceId: string | null, stage: StageId, enabled: boolean) {
  return useQuery<T>({
    queryKey: getStageDataQueryKey(workspaceId, stage),
    queryFn: () => fetchStageData<T>(workspaceId as string, stage),
    enabled: !!workspaceId && enabled,
    staleTime: Number.POSITIVE_INFINITY,
  });
}
