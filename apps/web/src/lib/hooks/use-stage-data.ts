"use client";

import type { StageId } from "@causal-ssm/api-types";
import type { StageRunStatus } from "./use-run-events";
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

/**
 * Fetch stage data with optional live polling.
 *
 * When `status` is "running", enables 3-second polling to pick up partial
 * traces written by the pipeline. When completed, fetches once and caches
 * indefinitely (staleTime: Infinity).
 */
export function useStageData<T>(
  runId: string | null,
  stage: StageId,
  enabled: boolean,
  status?: StageRunStatus,
) {
  const isRunning = status === "running";

  return useQuery<T>({
    queryKey: ["pipeline", runId, "stage", stage],
    queryFn: () => fetchStageData<T>(runId as string, stage),
    enabled: !!runId && (enabled || isRunning),
    staleTime: isRunning ? 0 : Number.POSITIVE_INFINITY,
    refetchInterval: isRunning ? 3_000 : false,
  });
}
