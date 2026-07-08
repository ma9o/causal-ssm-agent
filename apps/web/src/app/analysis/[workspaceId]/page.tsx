"use client";

import { AnalysisFeed } from "@/components/pipeline/analysis-feed";
import { getAnalysisManifest, getAnalysisManifestQueryKey } from "@/lib/api/analysis";
import { usePipelineStatus } from "@/lib/hooks/use-pipeline-status";
import { useRunEvents } from "@/lib/hooks/use-run-events";
import { TRANSITION_META } from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { use, useEffect, useMemo } from "react";

export default function AnalysisPage({ params }: { params: Promise<{ workspaceId: string }> }) {
  const { workspaceId } = use(params);
  const progress = usePipelineStatus(workspaceId);
  const manifestQuery = useQuery({
    queryKey: getAnalysisManifestQueryKey(workspaceId),
    queryFn: () => getAnalysisManifest(workspaceId),
    enabled: !!workspaceId,
    staleTime: Infinity,
    retry: false,
  });
  const manifest = manifestQuery.data;
  const manifestError = useMemo(() => {
    if (manifest) {
      return null;
    }

    if (!manifestQuery.error) {
      return null;
    }

    return manifestQuery.error.message;
  }, [manifest, manifestQuery.error]);

  useRunEvents(workspaceId, manifest?.transitionOrder);

  useEffect(() => {
    if (!progress) {
      document.title = "Starting... | nof1-causal-lab";
      return;
    }

    if (progress.isComplete) {
      document.title = "Analysis Complete | nof1-causal-lab";
      return;
    }

    if (progress.isFailed) {
      document.title = "Failed | nof1-causal-lab";
      return;
    }

    const completed = progress.transitionOrder.filter(
      (artifactId) => progress.artifacts[artifactId] === "completed",
    ).length;
    const currentLabels = progress.runningTransitions.map(
      (artifactId) => TRANSITION_META[artifactId].label,
    );
    const current = currentLabels.length > 0 ? currentLabels.join(", ") : null;

    document.title = current
      ? `(${completed}/${progress.transitionOrder.length}) ${current} | nof1-causal-lab`
      : `(${completed}/${progress.transitionOrder.length}) Running | nof1-causal-lab`;
  }, [progress]);

  const mainContent = manifestError ? (
    <div className="flex min-h-screen items-center justify-center px-4 py-10 sm:px-6">
      <div className="max-w-md space-y-3 rounded-lg border bg-card p-6 text-center">
        <h1 className="text-lg font-semibold">Workspace unavailable</h1>
        <p className="text-sm text-muted-foreground">{manifestError}</p>
        <Link
          href="/"
          className="inline-flex rounded-md border px-3 py-2 text-sm font-medium transition-colors hover:bg-secondary"
        >
          Return Home
        </Link>
      </div>
    </div>
  ) : (
    <AnalysisFeed
      workspaceId={workspaceId}
      question={manifest?.question}
      transitionRuns={manifest?.transitionRuns}
      progress={progress}
      readOnly={manifest?.readOnly ?? false}
    />
  );

  return mainContent;
}
