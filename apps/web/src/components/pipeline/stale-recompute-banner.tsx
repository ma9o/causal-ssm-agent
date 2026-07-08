"use client";

import { recomputeStaleArtifacts } from "@/lib/api/analysis";
import { getEpisodeProgressQueryKey } from "@/lib/hooks/use-run-events";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { useQueryClient } from "@tanstack/react-query";
import { Loader2, RefreshCw } from "lucide-react";
import { useCallback, useState } from "react";

/** Presentational banner — no hooks beyond local UI state. Used by stories too. */
export function StaleRecomputeBannerView({
  staleTransitionCount,
  recomputing,
  error,
  onRecompute,
}: {
  staleTransitionCount: number;
  recomputing: boolean;
  error?: string | null;
  onRecompute: () => void;
}) {
  return (
    <div
      role="status"
      className="mx-auto flex w-full max-w-[1600px] items-center justify-between gap-3 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3"
    >
      <div className="flex items-center gap-2 text-sm">
        <RefreshCw className="h-4 w-4 shrink-0 text-warning-foreground" />
        <span className="font-medium text-warning-foreground">
          {staleTransitionCount} artifact{staleTransitionCount === 1 ? "" : "s"} have stale results
        </span>
        <span className="hidden text-muted-foreground sm:inline">
          — inputs changed since they last ran.
        </span>
        {error && <span className="text-destructive">{error}</span>}
      </div>
      <button
        type="button"
        onClick={onRecompute}
        disabled={recomputing}
        className="flex shrink-0 items-center gap-1.5 rounded-md border border-warning/50 bg-background px-3 py-1.5 text-sm font-medium transition-colors hover:bg-warning/20 disabled:opacity-60"
      >
        {recomputing ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Recomputing...
          </>
        ) : (
          "Recompute"
        )}
      </button>
    </div>
  );
}

export function countStaleTransitions(progress: PipelineProgress): number {
  return progress.transitionOrder.filter(
    (artifactId) =>
      (progress.staleArtifactsByProducer[artifactId]?.length ?? 0) > 0 &&
      progress.artifacts[artifactId] !== "running",
  ).length;
}

/**
 * Shown when the machine reports stale artifacts while no auto-run is
 * active — moves made through other surfaces (facade calls, an LLM
 * navigator) leave staleness pending until a recompute is requested.
 */
export function StaleRecomputeBanner({
  workspaceId,
  progress,
}: {
  workspaceId: string;
  progress: PipelineProgress;
}) {
  const queryClient = useQueryClient();
  const [recomputing, setRecomputing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRecompute = useCallback(async () => {
    if (recomputing) return;
    setRecomputing(true);
    setError(null);
    try {
      await recomputeStaleArtifacts(workspaceId);
      // Refetch immediately so auto_running flips and the poll resumes.
      await queryClient.invalidateQueries({
        queryKey: getEpisodeProgressQueryKey(workspaceId),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Recompute failed");
    } finally {
      setRecomputing(false);
    }
  }, [queryClient, recomputing, workspaceId]);

  const staleTransitionCount = countStaleTransitions(progress);
  if (staleTransitionCount === 0 || progress.autoRunning) {
    return null;
  }

  return (
    <StaleRecomputeBannerView
      staleTransitionCount={staleTransitionCount}
      recomputing={recomputing}
      error={error}
      onRecompute={() => void handleRecompute()}
    />
  );
}
