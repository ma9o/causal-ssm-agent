import { Tooltip } from "@/components/ui/tooltip";
import { useExportMarkdown } from "@/lib/hooks/use-export-markdown";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Check, Copy, Download, Loader2, X } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

export function PipelineProgressBar({
  progress,
  sessionCode,
  runId,
}: {
  progress: PipelineProgress | undefined;
  sessionCode?: string;
  runId: string;
}) {
  const [copied, setCopied] = useState(false);

  const { data: session } = useQuery<{ question: string }>({
    queryKey: ["session", sessionCode],
    queryFn: async () => {
      const res = await fetch(`/api/sessions/${sessionCode}`);
      if (!res.ok) throw new Error("Session not found");
      return res.json();
    },
    enabled: !!sessionCode,
    staleTime: Number.POSITIVE_INFINITY,
  });

  const copyTimerRef = useRef<ReturnType<typeof setTimeout>>(null);
  const handleCopy = useCallback(() => {
    if (!sessionCode) return;
    navigator.clipboard.writeText(sessionCode);
    setCopied(true);
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    copyTimerRef.current = setTimeout(() => setCopied(false), 1500);
  }, [sessionCode]);
  useEffect(
    () => () => {
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    },
    [],
  );

  const { exportToMarkdown } = useExportMarkdown(runId);

  if (!progress) return null;

  const completed = STAGES.filter((s) => progress.stages[s.id] === "completed").length;

  return (
    <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-sm border-b px-4 py-2.5 sm:px-6 sm:py-3">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-1.5">
          <Link
            href="/"
            className="text-base font-semibold tracking-tight hover:opacity-80 transition-opacity"
          >
            Causal SSM Agent
          </Link>
          <div className="flex items-center gap-2">
            {sessionCode && (
              <button
                type="button"
                onClick={handleCopy}
                className="flex items-center gap-1 rounded border bg-secondary/50 px-2 py-0.5 font-mono text-xs tracking-widest text-muted-foreground transition-colors hover:bg-secondary"
                title="Copy session code"
              >
                {sessionCode.length > 8 ? "Signed in" : sessionCode}
                {copied ? <Check className="h-3 w-3 text-success" /> : <Copy className="h-3 w-3" />}
              </button>
            )}
            <span className="text-sm font-medium text-muted-foreground">
              {completed}/{STAGES.length} stages
            </span>
            {completed > 0 && (
              <Tooltip content={<span className="text-xs">Export as Markdown</span>}>
                <button
                  type="button"
                  onClick={exportToMarkdown}
                  className="flex items-center justify-center rounded border bg-secondary/50 p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                  title="Export report as Markdown"
                >
                  <Download className="h-3.5 w-3.5" />
                </button>
              </Tooltip>
            )}
          </div>
        </div>
        {session?.question && (
          <p className="text-sm text-muted-foreground mb-1.5">{session.question}</p>
        )}
        <div className="flex items-center gap-1.5">
          {STAGES.map((stage) => {
            const status = progress.stages[stage.id];
            const outcome = progress.stageOutcomes[stage.id];
            const isClickable = status !== "pending";

            const tooltipIcon =
              outcome === "fail" || status === "failed" ? (
                <X className="h-3 w-3 text-destructive" />
              ) : outcome === "warn" ? (
                <AlertTriangle className="h-3 w-3 text-warning" />
              ) : status === "completed" ? (
                <Check className="h-3 w-3 text-success" />
              ) : status === "running" ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : null;

            const tooltipSuffix =
              outcome === "fail" ? " (blocked)" : outcome === "warn" ? " (warning)" : "";

            const segmentColor =
              outcome === "fail" || status === "failed"
                ? "bg-destructive"
                : outcome === "warn"
                  ? "bg-warning"
                  : status === "completed"
                    ? "bg-success"
                    : status === "running"
                      ? "bg-primary animate-pulse-subtle"
                      : "bg-secondary";

            return (
              <Tooltip
                key={stage.id}
                triggerClassName="flex-1"
                content={
                  <div className="flex items-center gap-1.5 text-xs whitespace-nowrap">
                    {tooltipIcon}
                    <span>
                      {stage.number}. {stage.label}
                      {tooltipSuffix}
                    </span>
                  </div>
                }
              >
                <button
                  type="button"
                  disabled={!isClickable}
                  className="group relative w-full"
                  onClick={() => {
                    if (!isClickable) return;
                    document
                      .getElementById(stage.id)
                      ?.scrollIntoView({ behavior: "smooth", block: "start" });
                  }}
                >
                  <div
                    className={`h-1.5 rounded-full transition-all duration-500 ${segmentColor} ${isClickable ? "group-hover:opacity-80 cursor-pointer" : "cursor-default"}`}
                  />
                </button>
              </Tooltip>
            );
          })}
        </div>
      </div>
    </header>
  );
}
