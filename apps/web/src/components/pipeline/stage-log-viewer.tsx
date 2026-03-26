"use client";

import type { AnalysisStageRun } from "@/lib/api/analysis";
import type { StageId } from "@causal-ssm/api-types";
import { useStageLogs } from "@/lib/hooks/use-stage-logs";
import { Terminal } from "lucide-react";
import { useState } from "react";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { VirtualizedLogList } from "./virtualized-log-list";

export function StageLogViewer({
  workspaceId,
  stageId,
  status,
  stageRun,
}: {
  workspaceId: string;
  stageId: StageId;
  status: StageRunStatus;
  stageRun?: AnalysisStageRun | null;
}) {
  const {
    logs,
    bootstrapStatus,
    connectionState,
  } = useStageLogs(
    workspaceId,
    stageId,
    stageRun,
    status,
  );
  const [open, setOpen] = useState(false);

  if (logs.length === 0 && status !== "running") return null;

  const emptyMessage =
    bootstrapStatus === "pending"
      ? "Loading log backlog..."
      : bootstrapStatus === "error"
        ? "Failed to load historical logs."
      : connectionState === "error"
        ? "Live log stream unavailable."
        : connectionState === "connecting" || connectionState === "authenticating"
          ? "Connecting to live log stream..."
          : "Waiting for logs...";

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
      {status === "running" && connectionState === "error" && (
        <p className="mt-2 text-xs text-destructive">
          Live log stream disconnected. Prefect `logs/out` must be available for running-stage logs.
        </p>
      )}
      {open && (
        <VirtualizedLogList
          logs={logs}
          emptyMessage={emptyMessage}
          autoScroll={status === "running"}
          className="mt-2 bg-muted/30 p-2"
        />
      )}
    </div>
  );
}
