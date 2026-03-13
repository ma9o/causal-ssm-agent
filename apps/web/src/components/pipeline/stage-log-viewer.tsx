"use client";

import { type PrefectLogEntry, logLevelLabel, useStageLogs } from "@/lib/hooks/use-stage-logs";
import { cn } from "@/lib/utils";
import type { StageId } from "@causal-ssm/api-types";
import { Terminal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";

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
      <span className="shrink-0 text-muted-foreground/40 select-none">{ts}</span>
      <span className={cn("shrink-0 w-12 text-right", color)}>{level}</span>
      <span className={cn("break-all", entry.level >= 40 && "text-red-500")}>
        {entry.message}
      </span>
    </div>
  );
}

export function StageLogViewer({
  runId,
  stageId,
  status,
}: {
  runId: string;
  stageId: StageId;
  status: StageRunStatus;
}) {
  const logs = useStageLogs(runId, stageId, status);
  const [open, setOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll when new logs arrive while panel is open
  useEffect(() => {
    if (open && status === "running") {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs.length, open, status]);

  if (logs.length === 0 && status !== "running") return null;

  return (
    <div className="mt-3 border-t pt-3">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <Terminal className="h-3.5 w-3.5" />
        {open ? "Hide" : "Show"} logs
        {logs.length > 0 && (
          <span className="text-muted-foreground/50">({logs.length})</span>
        )}
      </button>
      {open && (
        <div className="mt-2 max-h-64 overflow-y-auto rounded-md bg-muted/30 p-2 font-mono text-[11px]">
          {logs.length === 0 ? (
            <p className="text-muted-foreground/50 text-center py-2">Waiting for logs...</p>
          ) : (
            logs.map((entry) => <LogLine key={entry.id} entry={entry} />)
          )}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
