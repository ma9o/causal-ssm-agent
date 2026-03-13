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
  params: Promise<{ code: string }>;
  searchParams: Promise<{ flowRunId?: string }>;
}) {
  const { code } = use(params);
  const { flowRunId } = use(searchParams);

  useRunEvents(code, flowRunId ?? null);
  const progress = usePipelineStatus(code);

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

  return <AnalysisFeed code={code} flowRunId={flowRunId} progress={progress} />;
}
