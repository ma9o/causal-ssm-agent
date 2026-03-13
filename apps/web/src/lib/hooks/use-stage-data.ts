"use client";

import type { StageId } from "@causal-ssm/api-types";
import { useQuery } from "@tanstack/react-query";
import { getStageResult } from "../api/endpoints";
import { isMockMode } from "../api/mock-provider";

async function fetchStageData<T>(code: string, stage: StageId): Promise<T> {
  let payload: unknown;

  if (isMockMode()) {
    const res = await fetch(`/api/results/${code}/${stage}`);
    if (!res.ok) throw new Error(`Mock data not found for ${stage}`);
    payload = await res.json();
  } else {
    payload = await getStageResult<unknown>(code, stage);
  }

  return payload as T;
}

/**
 * Fetch stage data once after completion and cache indefinitely.
 */
export function useStageData<T>(
  code: string | null,
  stage: StageId,
  enabled: boolean,
) {
  return useQuery<T>({
    queryKey: ["pipeline", code, "stage", stage],
    queryFn: () => fetchStageData<T>(code as string, stage),
    enabled: !!code && enabled,
    staleTime: Number.POSITIVE_INFINITY,
  });
}
