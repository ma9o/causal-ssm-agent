"use client";

import type { AnalysisStageRun } from "@/lib/api/analysis";
import {
  buildPrefectSubscriptionKey,
  buildStageLogScopePath,
  getStageLogScopeRefreshIntervalMs,
  getStageRuntimeInitialLogFlowRunIds,
  toStageRuntimeRef,
} from "@/lib/stage-observability";
import type { StageId } from "@causal-ssm/api-types";
import { useQuery } from "@tanstack/react-query";
import type { StageRunStatus } from "./pipeline-progress";

interface StageLogScopeResponse {
  flowRunIds?: string[];
}

export interface StageLogScopeResolution {
  runtime: ReturnType<typeof toStageRuntimeRef>;
  flowRunIds: string[];
  subscriptionKey: string;
}

export function useStageLogScope(
  workspaceId: string,
  stageId: StageId,
  stageRun: AnalysisStageRun | null | undefined,
  status: StageRunStatus,
): StageLogScopeResolution {
  const runtime = toStageRuntimeRef(stageRun);
  const initialFlowRunIds = getStageRuntimeInitialLogFlowRunIds(stageRun);
  const initialSignature = buildPrefectSubscriptionKey(initialFlowRunIds);
  const refreshInterval = getStageLogScopeRefreshIntervalMs(
    stageId,
    status === "running",
    runtime.stageSubflowRunId,
  );

  const { data } = useQuery({
    queryKey: [
      "analysis",
      workspaceId,
      "stage-log-scope",
      stageId,
      runtime.stageSubflowRunId,
      initialSignature,
    ] as const,
    queryFn: async () => {
      if (!runtime.stageSubflowRunId) {
        return initialFlowRunIds;
      }

      const response = await fetch(
        buildStageLogScopePath(workspaceId, stageId, runtime.stageSubflowRunId),
        { cache: "no-store" },
      );
      if (!response.ok) {
        return initialFlowRunIds;
      }

      const payload = (await response.json()) as StageLogScopeResponse;
      return (payload.flowRunIds ?? []).filter(
        (flowRunId): flowRunId is string => typeof flowRunId === "string" && flowRunId.trim().length > 0,
      );
    },
    enabled: refreshInterval !== false,
    initialData: initialFlowRunIds,
    initialDataUpdatedAt: 0,
    refetchInterval: refreshInterval,
    staleTime: 1000,
  });

  const flowRunIds = refreshInterval !== false ? (data ?? initialFlowRunIds) : initialFlowRunIds;
  return {
    runtime,
    flowRunIds,
    subscriptionKey: buildPrefectSubscriptionKey(flowRunIds),
  };
}
