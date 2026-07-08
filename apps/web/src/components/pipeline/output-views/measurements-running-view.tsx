"use client";

import { useExtractionState } from "@/lib/hooks/use-extraction-state";
import { cn } from "@/lib/utils";
import { CheckCircle2, Gauge, Loader2, XCircle } from "lucide-react";
import type { ExtractionWorkerRecord } from "@/lib/extraction-runtime";

const MAX_RPM = 450;

function WorkerGrid({ workers }: { workers: ExtractionWorkerRecord[] }) {
  if (workers.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-[3px]">
      {workers.map((w) => (
        <div
          key={w.worker_id}
          className={cn(
            "h-2.5 w-2.5 rounded-[2px] transition-colors duration-300",
            w.state === "completed" && "bg-emerald-500",
            w.state === "failed" && "bg-destructive",
            w.state === "running" && "bg-primary animate-pulse",
            w.state === "pending" && "bg-muted-foreground/20",
          )}
          title={`extract-chunk-${w.worker_id}: ${w.state}`}
        />
      ))}
    </div>
  );
}
function RpmGauge({ rpm, maxRpm = MAX_RPM }: { rpm: number; maxRpm?: number }) {
  const pct = Math.min(100, (rpm / maxRpm) * 100);
  const isHigh = pct > 80;
  const isMed = pct > 50;

  return (
    <div className="flex items-center gap-2 text-xs">
      <Gauge className="h-3.5 w-3.5 text-muted-foreground" />
      <div className="flex items-center gap-1.5">
        <div className="w-16 h-1.5 rounded-full bg-muted overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              isHigh ? "bg-red-500" : isMed ? "bg-amber-500" : "bg-emerald-500",
            )}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className={cn("tabular-nums", isHigh ? "text-red-500" : "text-muted-foreground")}>
          {rpm}/{maxRpm} rpm
        </span>
      </div>
    </div>
  );
}

/** Presentational component — no hooks, pure props. Used by stories too. */
export function MeasurementsRunningView({
  workers,
  failed,
  maxRpm = MAX_RPM,
  running,
  rpm = 0,
  total,
}: {
  workers: ExtractionWorkerRecord[];
  failed?: number;
  maxRpm?: number;
  running?: number;
  rpm?: number;
  total: number;
}) {
  const completed = workers.filter((w) => w.state === "completed").length;
  const failedCount = failed ?? workers.filter((w) => w.state === "failed").length;
  const runningCount = running ?? workers.filter((w) => w.state === "running").length;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-sm">
          {total > 0 ? (
            <>
              <span className="flex items-center gap-1.5 font-medium tabular-nums">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                {completed}
                <span className="text-muted-foreground font-normal">/</span>
                {total} done
              </span>
              {failedCount > 0 && (
                <span className="flex items-center gap-1 text-destructive">
                  <XCircle className="h-3.5 w-3.5" />
                  {failedCount} failed
                </span>
              )}
              {runningCount > 0 && (
                <span className="flex items-center gap-1 text-muted-foreground">
                  <CheckCircle2 className="h-3.5 w-3.5 text-primary" />
                  {runningCount} running
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

        {rpm > 0 && <RpmGauge rpm={rpm} maxRpm={maxRpm} />}
      </div>

      <WorkerGrid workers={workers} />
    </div>
  );
}

export default function MeasurementsRunningOutputView({ workspaceId }: { workspaceId: string }) {
  const { workers, summary, rpm, maxRpm } = useExtractionState(workspaceId);

  return (
    <MeasurementsRunningView
      workers={workers}
      total={summary.total}
      failed={summary.failed}
      running={summary.running}
      rpm={rpm}
      maxRpm={maxRpm}
    />
  );
}
