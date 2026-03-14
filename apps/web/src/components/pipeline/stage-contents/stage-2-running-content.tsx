"use client";

import { cn } from "@/lib/utils";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import {
  type Stage2Worker,
  useStage2Workers,
} from "@/lib/hooks/use-stage2-workers";
import { type PrefectLogEntry, logLevelLabel } from "@/lib/hooks/use-stage-logs";
import { CheckCircle2, Gauge, Loader2, XCircle } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

const LEVEL_COLORS: Record<number, string> = {
  10: "text-muted-foreground/50",
  20: "text-muted-foreground",
  30: "text-amber-500",
  40: "text-red-500",
  50: "text-red-600 font-semibold",
};

const MAX_RPM = 450;

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
  logs,
  rpm = 0,
}: {
  workers: Stage2Worker[];
  logs: PrefectLogEntry[];
  rpm?: number;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  const total = workers.length;
  const completed = workers.filter((w) => w.state === "completed").length;
  const failed = workers.filter((w) => w.state === "failed").length;
  const running = workers.filter((w) => w.state === "running").length;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  return (
    <div className="space-y-4">
      {/* Summary row */}
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

        {/* RPM gauge */}
        {rpm > 0 && <RpmGauge rpm={rpm} />}
      </div>

      {/* Worker grid — multi-row squares */}
      <WorkerGrid workers={workers} />

      {/* Log viewer */}
      <div className="max-h-64 overflow-y-auto rounded-md border border-border/50 bg-muted/20 p-3 font-mono text-[11px] leading-relaxed">
        {logs.length === 0 ? (
          <p className="text-muted-foreground/50 text-center py-4">
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

export default function Stage2RunningContent({
  userId,
  stageStatus,
  stageSubflowRunId,
}: {
  userId: string;
  stageStatus: StageRunStatus;
  stageSubflowRunId?: string | null;
}) {
  const { workers, logs } = useStage2Workers(userId, stageSubflowRunId ?? null, stageStatus);
  const rpm = useRpm(workers);

  return <Stage2RunningView workers={workers} logs={logs} rpm={rpm} />;
}
