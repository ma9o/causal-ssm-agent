"use client";

import { AnalysisFeed } from "@/components/pipeline/analysis-feed";
import {
  getAnalysisManifest,
  getAnalysisManifestQueryKey,
} from "@/lib/api/analysis";
import { hasStoppedStage } from "@/lib/hooks/pipeline-progress";
import { usePipelineStatus } from "@/lib/hooks/use-pipeline-status";
import { useRunEvents } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { use, useEffect, useMemo } from "react";

export default function AnalysisPage({
  params,
  searchParams,
}: {
  params: Promise<{ workspaceId: string }>;
  searchParams: Promise<{ rootFlowRunId?: string | string[] }>;
}) {
  const { workspaceId } = use(params);
  const rawBootstrapRootFlowRunId = use(searchParams).rootFlowRunId;
  const bootstrapRootFlowRunId = Array.isArray(rawBootstrapRootFlowRunId)
    ? rawBootstrapRootFlowRunId[0] ?? null
    : rawBootstrapRootFlowRunId ?? null;
  const progress = usePipelineStatus(workspaceId);
  const manifestQuery = useQuery({
    queryKey: getAnalysisManifestQueryKey(workspaceId, bootstrapRootFlowRunId),
    queryFn: () => getAnalysisManifest(workspaceId, bootstrapRootFlowRunId),
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

    if (manifestQuery.error.message.includes("API error 401")) {
      return "This analysis is only available in the browser session that started it.";
    }
    if (manifestQuery.error.message.includes("API error 403")) {
      return "Workspace access denied for this analysis.";
    }

    return manifestQuery.error.message;
  }, [manifest, manifestQuery.error]);

  useRunEvents(workspaceId, manifest?.rootFlowRunIds ?? [], manifest?.stages);

  useEffect(() => {
    if (!progress) {
      document.title = "Starting... | causal-ssm-agent";
      return;
    }

    if (progress.isComplete) {
      document.title = "Analysis Complete | causal-ssm-agent";
      return;
    }

    if (hasStoppedStage(progress)) {
      document.title = "Analysis Stopped | causal-ssm-agent";
      return;
    }

    if (progress.isFailed) {
      document.title = "Failed | causal-ssm-agent";
      return;
    }

    const completed = STAGES.filter((s) => progress.stages[s.id] === "completed").length;
    const current = progress.currentStage
      ? STAGES.find((s) => s.id === progress.currentStage)?.label
      : null;

    document.title = current
      ? `(${completed}/${STAGES.length}) ${current} | causal-ssm-agent`
      : `(${completed}/${STAGES.length}) Running | causal-ssm-agent`;
  }, [progress]);

  if (manifestError) {
    return (
      <div className="flex min-h-screen items-center justify-center px-4">
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
    );
  }

  return (
    <AnalysisFeed
      workspaceId={workspaceId}
      question={manifest?.question}
      stageRuns={manifest?.stages}
      progress={progress}
      latestRootFlowRunId={manifest?.latestRootFlowRunId}
    />
  );
}
