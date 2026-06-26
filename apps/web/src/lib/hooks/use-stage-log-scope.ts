"use client";

import type { AnalysisStageRun } from "@/lib/api/analysis";
import {
  buildStageLogScopeDescriptor,
  buildStageLogSubscriptionKey,
} from "@/lib/stage-observability";
import type { PrefectLogTimeWindow } from "@/lib/prefect-log-client";
import type { StageId } from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";
import type { StageRunStatus } from "./pipeline-progress";

interface StageLogScopeResponse {
  flowRunIds?: string[];
}

export interface StageLogScopeResolution {
  runtime: ReturnType<typeof buildStageLogScopeDescriptor>["runtime"];
  flowRunIds: string[];
  timeWindow: PrefectLogTimeWindow;
  subscriptionKey: string;
}

export function useStageLogScope(
  workspaceId: string,
  stageId: StageId,
  stageRun: AnalysisStageRun | null | undefined,
  status: StageRunStatus,
): StageLogScopeResolution {
  const descriptor = buildStageLogScopeDescriptor(workspaceId, stageId, stageRun, status);
  const { runtime, initialFlowRunIds, timeWindow } = descriptor;
  const initialSignature = buildStageLogSubscriptionKey(initialFlowRunIds, timeWindow);

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
      if (descriptor.refresh === false) {
        return initialFlowRunIds;
      }

      const response = await fetch(descriptor.refresh.path, { cache: "no-store" });
      if (!response.ok) {
        return initialFlowRunIds;
      }

      const payload = (await response.json()) as StageLogScopeResponse;
      return (payload.flowRunIds ?? []).filter(
        (flowRunId): flowRunId is string =>
          typeof flowRunId === "string" && flowRunId.trim().length > 0,
      );
    },
    enabled: descriptor.refresh !== false,
    initialData: initialFlowRunIds,
    initialDataUpdatedAt: 0,
    refetchInterval: descriptor.refresh === false ? false : descriptor.refresh.intervalMs,
    staleTime: 1000,
  });

  const flowRunIds = descriptor.refresh !== false ? (data ?? initialFlowRunIds) : initialFlowRunIds;
  return {
    runtime,
    flowRunIds,
    timeWindow,
    subscriptionKey: buildStageLogSubscriptionKey(flowRunIds, timeWindow),
  };
}
