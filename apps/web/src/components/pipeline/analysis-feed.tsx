"use client";

import { BackToTop } from "@/components/back-to-top";
import { Skeleton } from "@/components/ui/skeleton";
import type { AnalysisTransitionRuns } from "@/lib/api/analysis";
import { WorkspaceViewProvider } from "@/lib/contexts/workspace-view-context";
import { useKeyboardNav } from "@/lib/hooks/use-keyboard-nav";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { TRANSITION_META } from "@nof1-causal-lab/api-types";
import { Loader2 } from "lucide-react";
import { useMemo } from "react";
import { ActiveTransitionsIndicator } from "./active-transitions-indicator";
import { LazyOutputMount } from "./lazy-output-mount";
import { CompletedOutputsNotification } from "./completed-outputs-notification";
import { PipelineProgressBar } from "./progress-bar";
import { OutputSectionRouter } from "./output-section-router";
import { StaleRecomputeBanner } from "./stale-recompute-banner";

function FeedContent({
  workspaceId,
  transitionRuns,
  question,
  progress,
  readOnly,
}: {
  workspaceId: string;
  transitionRuns?: AnalysisTransitionRuns;
  question?: string;
  progress: PipelineProgress;
  readOnly: boolean;
}) {
  const visibleArtifactIds = useMemo(
    () =>
      progress.transitionOrder.filter((artifactId) => progress.artifacts[artifactId] !== "pending"),
    [progress],
  );
  useKeyboardNav(visibleArtifactIds);

  const visibleOutputs = visibleArtifactIds.map((artifactId) => TRANSITION_META[artifactId]);

  return (
    <div>
      <PipelineProgressBar progress={progress} question={question} workspaceId={workspaceId} />
      <div className="space-y-4 px-4 py-6 sm:space-y-6 sm:px-6 lg:px-10 2xl:px-12">
        {!readOnly && <StaleRecomputeBanner workspaceId={workspaceId} progress={progress} />}
        {visibleOutputs.map((output) => (
          <LazyOutputMount key={output.id} output={output}>
            <OutputSectionRouter
              output={output}
              workspaceId={workspaceId}
              transitionRun={transitionRuns?.[output.id]}
              status={progress.artifacts[output.id]}
              timing={progress.timings[output.id]}
              errorMessage={progress.transitionErrors[output.id]}
              staleArtifactIds={progress.staleArtifactsByProducer[output.id]}
            />
          </LazyOutputMount>
        ))}
        {!progress.isComplete && (
          <div className="mx-auto w-full max-w-[1600px]">
            <ActiveTransitionsIndicator artifactIds={progress.runningTransitions} />
          </div>
        )}
      </div>
      <CompletedOutputsNotification progress={progress} />
      <BackToTop />
    </div>
  );
}

export function AnalysisFeed({
  workspaceId,
  transitionRuns,
  question,
  progress,
  readOnly = false,
}: {
  workspaceId: string;
  transitionRuns?: AnalysisTransitionRuns;
  question?: string;
  progress: PipelineProgress | undefined;
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
    <WorkspaceViewProvider readOnly={readOnly}>
      <FeedContent
        workspaceId={workspaceId}
        transitionRuns={transitionRuns}
        question={question}
        progress={progress}
        readOnly={readOnly}
      />
    </WorkspaceViewProvider>
  );
}
