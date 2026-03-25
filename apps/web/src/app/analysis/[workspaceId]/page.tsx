"use client";

import { AnalysisFeed } from "@/components/pipeline/analysis-feed";
import {
  getAnalysisManifest,
  getAnalysisManifestQueryKey,
  unlockWorkspace,
} from "@/lib/api/analysis";
import { getIdentity, setIdentity } from "@/lib/identity";
import { hasStoppedStage } from "@/lib/hooks/pipeline-progress";
import { usePipelineStatus } from "@/lib/hooks/use-pipeline-status";
import { useRunEvents } from "@/lib/hooks/use-run-events";
import { getSharedWorkspaceAccessCode } from "@/lib/resume-key";
import { STAGES } from "@causal-ssm/api-types";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { use, useEffect, useMemo, useState } from "react";

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
  const [workspaceReady, setWorkspaceReady] = useState(false);
  const [unlockError, setUnlockError] = useState<string | null>(null);
  const progress = usePipelineStatus(workspaceId);
  const manifestQuery = useQuery({
    queryKey: getAnalysisManifestQueryKey(workspaceId, bootstrapRootFlowRunId),
    queryFn: () => getAnalysisManifest(workspaceId, bootstrapRootFlowRunId),
    enabled: workspaceReady && !!workspaceId,
    staleTime: Infinity,
    retry: false,
  });
  const manifest = manifestQuery.data;
  const manifestError = useMemo(() => {
    if (manifest) {
      return null;
    }

    if (!manifestQuery.error) {
      return unlockError;
    }

    if (manifestQuery.error.message.includes("API error 401")) {
      return "Workspace locked. Return to the home page and enter the full resume key.";
    }
    if (manifestQuery.error.message.includes("API error 403")) {
      return "Workspace access denied. Check that the resume key matches this workspace.";
    }

    return manifestQuery.error.message;
  }, [manifest, manifestQuery.error, unlockError]);

  useEffect(() => {
    let cancelled = false;

    const storedIdentity = getIdentity();
    const accessCode =
      getSharedWorkspaceAccessCode(workspaceId) ??
      (storedIdentity?.workspaceId === workspaceId ? storedIdentity.accessCode : null);

    if (!accessCode) {
      setUnlockError(null);
      setWorkspaceReady(true);
      return () => {
        cancelled = true;
      };
    }

    setWorkspaceReady(false);
    setUnlockError(null);

    void unlockWorkspace(workspaceId, accessCode)
      .then(() => {
        if (cancelled) return;
        setIdentity({ workspaceId, accessCode, kind: storedIdentity?.kind ?? "anonymous" });
        setWorkspaceReady(true);
      })
      .catch((error) => {
        if (cancelled) return;
        setUnlockError(error instanceof Error ? error.message : "Failed to unlock workspace.");
        setWorkspaceReady(true);
      });

    return () => {
      cancelled = true;
    };
  }, [workspaceId]);

  useRunEvents(workspaceId, manifest?.rootFlowRunIds ?? [], manifest?.stages);

  useEffect(() => {
    if (!progress) {
      document.title = "Starting... | Causal Inference Pipeline";
      return;
    }

    if (progress.isComplete) {
      document.title = "Analysis Complete | Causal Inference Pipeline";
      return;
    }

    if (hasStoppedStage(progress)) {
      document.title = "Analysis Stopped | Causal Inference Pipeline";
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

  if (!workspaceReady) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-20">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <div className="text-center space-y-1">
          <p className="text-sm font-medium text-muted-foreground">Unlocking workspace...</p>
          <p className="text-xs text-muted-foreground/60">
            Restoring access before loading the analysis
          </p>
        </div>
      </div>
    );
  }

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
