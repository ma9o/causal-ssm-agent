"use client";

import { AnalysisFeed } from "@/components/pipeline/analysis-feed";
import {
  getAnalysisManifest,
  getAnalysisManifestQueryKey,
} from "@/lib/api/analysis";
import { usePipelineStatus } from "@/lib/hooks/use-pipeline-status";
import { useRunEvents } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import { useQuery } from "@tanstack/react-query";
import { use, useEffect } from "react";

export default function AnalysisPage({
  params,
  searchParams,
}: {
  params: Promise<{ userId: string }>;
  searchParams: Promise<{ rootFlowRunId?: string | string[] }>;
}) {
  const { userId } = use(params);
  const rawBootstrapRootFlowRunId = use(searchParams).rootFlowRunId;
  const bootstrapRootFlowRunId = Array.isArray(rawBootstrapRootFlowRunId)
    ? rawBootstrapRootFlowRunId[0] ?? null
    : rawBootstrapRootFlowRunId ?? null;
  const progress = usePipelineStatus(userId);
  const { data: manifest } = useQuery({
    queryKey: getAnalysisManifestQueryKey(userId, bootstrapRootFlowRunId),
    queryFn: () => getAnalysisManifest(userId, bootstrapRootFlowRunId),
    enabled: !!userId,
    refetchInterval: progress && !progress.isComplete && !progress.isFailed ? 3000 : false,
    staleTime: progress && !progress.isComplete && !progress.isFailed ? 1000 : Infinity,
  });

  useRunEvents(userId, manifest?.rootFlowRunIds ?? [], manifest?.stages);

  // Dynamic document title reflecting pipeline state
  useEffect(() => {
    if (!progress) {
      document.title = "Starting... | Causal Inference Pipeline";
      return;
    }

    if (progress.isComplete) {
      document.title = "Analysis Complete | Causal Inference Pipeline";
      return;
    }

    if (progress.isFailed) {
      document.title = "Failed | Causal Inference Pipeline";
      return;
    }

    const completed = STAGES.filter((s) => progress.stages[s.id] === "completed").length;
    const current = progress.currentStage
      ? STAGES.find((s) => s.id === progress.currentStage)?.label
      : null;

    document.title = current
      ? `(${completed}/${STAGES.length}) ${current} | Causal Inference Pipeline`
      : `(${completed}/${STAGES.length}) Running | Causal Inference Pipeline`;
  }, [progress]);

  return (
    <AnalysisFeed
      userId={userId}
      question={manifest?.question}
      stageRuns={manifest?.stages}
      progress={progress}
      latestRootFlowRunId={manifest?.latestRootFlowRunId}
    />
  );
}
