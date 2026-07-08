import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { TRANSITION_META } from "@nof1-causal-lab/api-types";
import { Check, Loader2, RefreshCw, X } from "lucide-react";
import Link from "next/link";

function formatWorkspaceIdBadge(workspaceId: string): string {
  if (workspaceId.length <= 18) return workspaceId;
  return `${workspaceId.slice(0, 8)}...${workspaceId.slice(-6)}`;
}

export function PipelineProgressBar({
  progress,
  question,
  workspaceId,
}: {
  progress: PipelineProgress | undefined;
  question?: string;
  workspaceId: string;
}) {
  if (!progress) return null;

  const completed = progress.transitionOrder.filter(
    (artifactId) => progress.artifacts[artifactId] === "completed",
  ).length;

  return (
    <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-sm border-b px-4 py-2.5 sm:px-6 sm:py-3 lg:px-10 2xl:px-12">
      <div className="mx-auto w-full max-w-[1600px]">
        <div className="flex items-center justify-between mb-1.5">
          <Link
            href="/"
            className="text-base font-semibold tracking-tight hover:opacity-80 transition-opacity"
          >
            N-of-1 Causal Lab
          </Link>
          <div className="flex items-center gap-2">
            {workspaceId && (
              <span
                className="rounded border bg-secondary/50 px-2 py-0.5 font-mono text-xs tracking-widest text-muted-foreground"
                title="Workspace ID"
              >
                {formatWorkspaceIdBadge(workspaceId)}
              </span>
            )}
            <span className="text-sm font-medium text-muted-foreground">
              {completed}/{progress.transitionOrder.length} artifacts
            </span>
          </div>
        </div>
        {question && <p className="text-sm text-muted-foreground mb-1.5">{question}</p>}
        <div className="flex items-center gap-1.5">
          {progress.transitionOrder.map((artifactId) => {
            const output = TRANSITION_META[artifactId];
            const status = progress.artifacts[output.id];
            const stale =
              (progress.staleArtifactsByProducer[output.id]?.length ?? 0) > 0 &&
              status !== "running";
            const isClickable = status !== "pending";

            const tooltipIcon =
              status === "failed" ? (
                <X className="h-3 w-3 text-destructive" />
              ) : stale ? (
                <RefreshCw className="h-3 w-3 text-warning" />
              ) : status === "completed" ? (
                <Check className="h-3 w-3 text-success" />
              ) : status === "running" ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : null;

            const tooltipSuffix =
              status === "failed" ? " (execution failed)" : stale ? " (stale)" : "";

            const segmentColor =
              status === "failed"
                ? "bg-destructive"
                : stale
                  ? "bg-warning"
                  : status === "completed"
                    ? "bg-success"
                    : status === "running"
                      ? "bg-primary animate-pulse-subtle"
                      : "bg-secondary";

            return (
              <Tooltip key={output.id}>
                <TooltipTrigger
                  className="group relative flex-1"
                  disabled={!isClickable}
                  onClick={() => {
                    if (!isClickable) return;
                    document
                      .getElementById(output.id)
                      ?.scrollIntoView({ behavior: "smooth", block: "start" });
                  }}
                >
                  <div
                    className={`h-1.5 rounded-full transition-all duration-500 ${segmentColor} ${isClickable ? "group-hover:opacity-80 cursor-pointer" : "cursor-default"}`}
                  />
                </TooltipTrigger>
                <TooltipContent>
                  <div className="flex items-center gap-1.5 text-xs whitespace-nowrap">
                    {tooltipIcon}
                    <span>
                      {output.label}
                      {tooltipSuffix}
                    </span>
                  </div>
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      </div>
    </header>
  );
}
