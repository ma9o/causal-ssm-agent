"use client";

import { BackToTop } from "@/components/back-to-top";
import { Skeleton } from "@/components/ui/skeleton";
import { useKeyboardNav } from "@/lib/hooks/use-keyboard-nav";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import { Loader2 } from "lucide-react";
import { useMemo } from "react";
import { ActiveStageIndicator } from "./active-stage-indicator";
import { NewStagesNotification } from "./new-stages-notification";
import { PipelineProgressBar } from "./progress-bar";
import { StageSectionRouter } from "./stage-section-router";

export function AnalysisFeed({
  runId,
  progress,
  sessionCode,
}: {
  runId: string;
  progress: PipelineProgress | undefined;
  sessionCode?: string;
}) {
  const visibleStageIds = useMemo(
    () =>
      progress
        ? STAGES.filter((s) => progress.stages[s.id] !== "pending").map((s) => s.id)
        : [],
    [progress],
  );
  useKeyboardNav(visibleStageIds);

  if (!progress) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-20">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <div className="text-center space-y-1">
          <p className="text-sm font-medium text-muted-foreground">
            Waiting for pipeline to start...
          </p>
          <p className="text-xs text-muted-foreground/60">
            This usually takes a few seconds
          </p>
        </div>
        <div className="w-full max-w-md space-y-3 mt-4">
          <Skeleton className="h-4 w-3/4 mx-auto" />
          <Skeleton className="h-4 w-1/2 mx-auto" />
        </div>
      </div>
    );
  }

  const visibleStages = STAGES.filter((s) => progress.stages[s.id] !== "pending");

  return (
    <div>
      <PipelineProgressBar progress={progress} sessionCode={sessionCode} />
      <div className="space-y-4 px-4 py-6 sm:space-y-6 sm:px-6">
        {visibleStages.map((stage) => (
          <StageSectionRouter
            key={stage.id}
            stage={stage}
            runId={runId}
            status={progress.stages[stage.id]}
            timing={progress.timings[stage.id]}
          />
        ))}
        {!progress.isComplete && <div className="max-w-6xl mx-auto"><ActiveStageIndicator stageId={progress.currentStage} /></div>}
      </div>
      <NewStagesNotification progress={progress} />
      <BackToTop />
    </div>
  );
}
