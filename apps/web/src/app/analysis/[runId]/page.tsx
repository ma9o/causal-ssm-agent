"use client";

import { AnalysisFeed } from "@/components/pipeline/analysis-feed";
import { usePipelineStatus } from "@/lib/hooks/use-pipeline-status";
import { useRunEvents } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import { use, useEffect } from "react";

export default function AnalysisPage({
  params,
  searchParams,
}: {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{ code?: string }>;
}) {
  const { runId } = use(params);
  const { code: sessionCode } = use(searchParams);

  useRunEvents(runId);
  const progress = usePipelineStatus(runId);

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

  return <AnalysisFeed runId={runId} progress={progress} sessionCode={sessionCode} />;
}
