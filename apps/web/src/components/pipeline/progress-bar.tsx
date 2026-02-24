import { Badge } from "@/components/ui/badge";
import { Tooltip } from "@/components/ui/tooltip";
import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import { Check, Copy, Loader2, X } from "lucide-react";
import { motion } from "motion/react";
import Link from "next/link";
import { useCallback, useState } from "react";

export function PipelineProgressBar({
  progress,
  sessionCode,
}: {
  progress: PipelineProgress | undefined;
  sessionCode?: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    if (!sessionCode) return;
    navigator.clipboard.writeText(sessionCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [sessionCode]);

  if (!progress) return null;

  const completed = STAGES.filter((s) => progress.stages[s.id] === "completed").length;
  const hasGateFailure = Object.values(progress.gateFailures).some(Boolean);
  const hasGateOverride = Object.values(progress.gateOverrides).some(Boolean);

  return (
    <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-sm border-b px-4 py-2.5 sm:px-6 sm:py-3">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-1.5">
          <Link
            href="/"
            className="text-base font-semibold tracking-tight hover:opacity-80 transition-opacity"
          >
            Causal Inference Pipeline
          </Link>
          <div className="flex items-center gap-2">
            {sessionCode && (
              <button
                type="button"
                onClick={handleCopy}
                className="flex items-center gap-1 rounded border bg-secondary/50 px-2 py-0.5 font-mono text-xs tracking-widest text-muted-foreground transition-colors hover:bg-secondary"
                title="Copy session code"
              >
                {sessionCode}
                {copied ? (
                  <Check className="h-3 w-3 text-success" />
                ) : (
                  <Copy className="h-3 w-3" />
                )}
              </button>
            )}
            <span className="text-sm font-medium text-muted-foreground">
              {completed}/{STAGES.length} stages
            </span>
            {progress.isComplete && !hasGateFailure && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
              >
                <Badge variant="success">Complete</Badge>
              </motion.span>
            )}
            {(progress.isFailed || hasGateFailure) && (
              <Badge variant="destructive">
                {hasGateFailure ? "Blocked" : "Failed"}
              </Badge>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {STAGES.map((stage) => {
            const status = progress.stages[stage.id];
            const isGateFailed = progress.gateFailures[stage.id] ?? false;
            const isGateOverridden = progress.gateOverrides[stage.id] ?? false;
            const isClickable = status !== "pending";

            const tooltipIcon = isGateFailed || status === "failed" || isGateOverridden
              ? <X className="h-3 w-3 text-destructive" />
              : status === "completed"
                  ? <Check className="h-3 w-3 text-success" />
                  : status === "running"
                    ? <Loader2 className="h-3 w-3 animate-spin" />
                    : null;

            const tooltipSuffix = isGateFailed ? " (blocked)" : isGateOverridden ? " (overridden)" : "";

            const segmentColor = isGateFailed || status === "failed" || isGateOverridden
              ? "bg-destructive"
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
                      className={`h-1 rounded-full transition-all duration-500 ${segmentColor} ${isClickable ? "group-hover:opacity-80 cursor-pointer" : "cursor-default"}`}
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
