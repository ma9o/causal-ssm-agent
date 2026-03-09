"use client";

import { cn } from "@/lib/utils/cn";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import {
  type Stage2Worker,
  useStage2Workers,
} from "@/lib/hooks/use-stage2-workers";
import { type PrefectLogEntry, logLevelLabel } from "@/lib/hooks/use-stage-logs";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { useEffect, useRef } from "react";

const LEVEL_COLORS: Record<number, string> = {
  10: "text-muted-foreground/50",
  20: "text-muted-foreground",
  30: "text-amber-500",
  40: "text-red-500",
  50: "text-red-600 font-semibold",
};

function LogLine({ entry }: { entry: PrefectLogEntry }) {
  const ts = new Date(entry.timestamp).toLocaleTimeString();
  const level = logLevelLabel(entry.level);
  const color = LEVEL_COLORS[entry.level] ?? "text-muted-foreground";

  return (
    <div className="flex gap-2 leading-5 hover:bg-muted/30">
      <span className="shrink-0 text-muted-foreground/40 select-none">
        {ts}
      </span>
      <span className={cn("shrink-0 w-12 text-right", color)}>{level}</span>
      <span className={cn("break-all", entry.level >= 40 && "text-red-500")}>
        {entry.message}
      </span>
    </div>
  );
}

function WorkerSegments({ workers }: { workers: Stage2Worker[] }) {
  if (workers.length === 0) return null;

  return (
    <div className="flex gap-0.5 h-2 w-full rounded-full overflow-hidden bg-muted">
      {workers.map((w) => (
        <div
          key={w.id}
          className={cn(
            "h-full flex-1 transition-colors duration-500",
            w.state === "completed" && "bg-emerald-500",
            w.state === "failed" && "bg-destructive",
            w.state === "running" && "bg-primary animate-pulse-subtle",
            w.state === "pending" && "bg-muted",
          )}
          title={`${w.name}: ${w.state}`}
        />
      ))}
    </div>
  );
}

export default function Stage2RunningContent({
  runId,
  stageStatus,
}: {
  runId: string;
  stageStatus: StageRunStatus;
}) {
  const { workers, logs } = useStage2Workers(runId, stageStatus);
  const bottomRef = useRef<HTMLDivElement>(null);

  const total = workers.length;
  const completed = workers.filter((w) => w.state === "completed").length;
  const failed = workers.filter((w) => w.state === "failed").length;
  const running = workers.filter((w) => w.state === "running").length;
  const done = completed + failed;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  return (
    <div className="space-y-3">
      {/* Summary */}
      <div className="flex items-center gap-3 text-sm">
        {total > 0 ? (
          <>
            <span className="flex items-center gap-1.5 font-medium">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
              {done}/{total} workers
            </span>
            {completed > 0 && (
              <span className="flex items-center gap-1 text-emerald-600">
                <CheckCircle2 className="h-3.5 w-3.5" />
                {completed}
              </span>
            )}
            {running > 0 && (
              <span className="text-muted-foreground">
                {running} running
              </span>
            )}
            {failed > 0 && (
              <span className="flex items-center gap-1 text-destructive">
                <XCircle className="h-3.5 w-3.5" />
                {failed}
              </span>
            )}
          </>
        ) : (
          <span className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Starting extraction workers...
          </span>
        )}
      </div>

      {/* Segmented progress bar */}
      <WorkerSegments workers={workers} />

      {/* Log viewer */}
      <div className="max-h-64 overflow-y-auto rounded-md bg-muted/30 p-2 font-mono text-[11px]">
        {logs.length === 0 ? (
          <p className="text-muted-foreground/50 text-center py-2">
            Waiting for worker logs...
          </p>
        ) : (
          logs.map((entry) => <LogLine key={entry.id} entry={entry} />)
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
