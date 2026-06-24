"use client";

import { BackToTop } from "@/components/back-to-top";
import { Skeleton } from "@/components/ui/skeleton";
import type { AnalysisStageRuns } from "@/lib/api/analysis";
import { RefinementProvider, useRefinement } from "@/lib/contexts/refinement-context";
import { useKeyboardNav } from "@/lib/hooks/use-keyboard-nav";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@nof1-causal-lab/api-types";
import { Loader2 } from "lucide-react";
import { useMemo } from "react";
import { ActiveStageIndicator } from "./active-stage-indicator";
import { InvalidationWarningModal } from "./invalidation-warning-modal";
import { LazyStageMount } from "./lazy-stage-mount";
import { NewStagesNotification } from "./new-stages-notification";
import { PipelineProgressBar } from "./progress-bar";
import { ResumeButton } from "./resume-button";
import { StageSectionRouter } from "./stage-section-router";

function FeedContent({
  workspaceId,
  stageRuns,
  question,
  progress,
  latestRootFlowRunId,
}: {
  workspaceId: string;
  stageRuns?: AnalysisStageRuns;
  question?: string;
  progress: PipelineProgress;
  latestRootFlowRunId?: string | null;
}) {
  const { refiningStageId, settled } = useRefinement();

  const visibleStageIds = useMemo(
    () => STAGES.filter((s) => progress.stages[s.id] !== "pending").map((s) => s.id),
    [progress],
  );
  useKeyboardNav(visibleStageIds);

  const visibleStages = STAGES.filter((s) => progress.stages[s.id] !== "pending");

  return (
    <div>
      <PipelineProgressBar progress={progress} question={question} workspaceId={workspaceId} />
      <div className="space-y-4 px-4 py-6 sm:space-y-6 sm:px-6 lg:px-10 2xl:px-12">
        {visibleStages.map((stage) => (
          <LazyStageMount key={stage.id} stage={stage}>
            <StageSectionRouter
              stage={stage}
              workspaceId={workspaceId}
              stageRun={stageRuns?.[stage.id]}
              status={progress.stages[stage.id]}
              timing={progress.timings[stage.id]}
            />
          </LazyStageMount>
        ))}
        {!progress.isComplete && (
          <div className="mx-auto w-full max-w-[1600px]">
            <ActiveStageIndicator stageId={progress.currentStage} />
          </div>
        )}
        {refiningStageId && settled && (
          <ResumeButton
            workspaceId={workspaceId}
            stageId={refiningStageId}
            rootFlowRunId={latestRootFlowRunId}
          />
        )}
      </div>
      <InvalidationWarningModal />
      <NewStagesNotification progress={progress} />
      <BackToTop />
    </div>
  );
}

export function AnalysisFeed({
  workspaceId,
  stageRuns,
  question,
  progress,
  latestRootFlowRunId,
  readOnly = false,
}: {
  workspaceId: string;
  stageRuns?: AnalysisStageRuns;
  question?: string;
  progress: PipelineProgress | undefined;
  latestRootFlowRunId?: string | null;
  readOnly?: boolean;
}) {
  if (!progress) {
    return (
      <div className="flex min-h-[50vh] flex-col items-center justify-center gap-4 px-4 py-20 sm:px-6">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <div className="space-y-1 text-center">
          <p className="text-sm font-medium text-muted-foreground">
            Waiting for pipeline to start...
          </p>
          <p className="text-xs text-muted-foreground/60">This usually takes a few seconds</p>
        </div>
        <div className="mt-4 w-full max-w-md space-y-3">
          <Skeleton className="mx-auto h-4 w-3/4" />
          <Skeleton className="mx-auto h-4 w-1/2" />
        </div>
      </div>
    );
  }

  return (
    <RefinementProvider readOnly={readOnly}>
      <FeedContent
        workspaceId={workspaceId}
        stageRuns={stageRuns}
        question={question}
        progress={progress}
        latestRootFlowRunId={latestRootFlowRunId}
      />
    </RefinementProvider>
  );
}
