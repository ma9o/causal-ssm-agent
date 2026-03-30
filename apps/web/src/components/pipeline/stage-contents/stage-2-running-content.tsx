"use client";

import type { AnalysisStageRun } from "@/lib/api/analysis";
import { cn } from "@/lib/utils";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import {
  type Stage2Worker,
  useStage2Workers,
} from "@/lib/hooks/use-stage2-workers";
import { CheckCircle2, Gauge, Loader2, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const MAX_RPM = 450;

function WorkerGrid({ workers }: { workers: Stage2Worker[] }) {
  if (workers.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-[3px]">
      {workers.map((w) => (
        <div
          key={w.id}
          className={cn(
            "h-2.5 w-2.5 rounded-[2px] transition-colors duration-300",
            w.state === "completed" && "bg-emerald-500",
            w.state === "failed" && "bg-destructive",
            w.state === "running" && "bg-primary animate-pulse",
            w.state === "pending" && "bg-muted-foreground/20",
          )}
          title={`${w.name}: ${w.state}`}
        />
      ))}
    </div>
  );
}

function useRpm(workers: Stage2Worker[]): number {
  // Periodic tick so the 60s window slides even when no new workers complete
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 5_000);
    return () => clearInterval(id);
  }, []);

  return useMemo(() => {
    const now = Date.now();
    const windowMs = 60_000;
    let totalCalls = 0;
    for (const w of workers) {
      if (w.completedAt && now - w.completedAt < windowMs && w.nLlmCalls) {
        totalCalls += w.nLlmCalls;
      }
    }
    // Rolling 60s count = RPM (matches OpenRouter's sliding window)
    return totalCalls;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workers, tick]);
}

function RpmGauge({ rpm }: { rpm: number }) {
  const pct = Math.min(100, (rpm / MAX_RPM) * 100);
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
        <span className={cn(
          "tabular-nums",
          isHigh ? "text-red-500" : "text-muted-foreground",
        )}>
          {rpm}/{MAX_RPM} rpm
        </span>
      </div>
    </div>
  );
}

/** Presentational component — no hooks, pure props. Used by stories too. */
export function Stage2RunningView({
  workers,
  rpm = 0,
}: {
  workers: Stage2Worker[];
  rpm?: number;
}) {
  const total = workers.length;
  const completed = workers.filter((w) => w.state === "completed").length;
  const failed = workers.filter((w) => w.state === "failed").length;
  const running = workers.filter((w) => w.state === "running").length;

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
              {failed > 0 && (
                <span className="flex items-center gap-1 text-destructive">
                  <XCircle className="h-3.5 w-3.5" />
                  {failed} failed
                </span>
              )}
              {running > 0 && (
                <span className="flex items-center gap-1 text-muted-foreground">
                  <CheckCircle2 className="h-3.5 w-3.5 text-primary" />
                  {running} running
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

        {rpm > 0 && <RpmGauge rpm={rpm} />}
      </div>

      <WorkerGrid workers={workers} />
    </div>
  );
}

export default function Stage2RunningContent({
  workspaceId,
  stageStatus,
  stageRun,
}: {
  workspaceId: string;
  stageStatus: StageRunStatus;
  stageRun?: AnalysisStageRun | null;
}) {
  const { workers } = useStage2Workers(workspaceId, stageRun, stageStatus);
  const rpm = useRpm(workers);

  return <Stage2RunningView workers={workers} rpm={rpm} />;
}
