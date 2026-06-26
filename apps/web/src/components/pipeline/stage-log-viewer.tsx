"use client";

import type { PrefectLogEntry } from "@/lib/prefect-log-client";
import type { PrefectSocketConnectionState } from "@/lib/hooks/use-prefect-socket";
import { Terminal } from "lucide-react";
import { useMemo, useState } from "react";
import type { QueryStatus } from "@tanstack/react-query";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { VirtualizedLogList } from "./virtualized-log-list";

const MAX_DISPLAY_LOGS = 1_000;

/** Presentational log view — no data-fetching, fully drivable from props. */
export function StageLogView({
  logs,
  status,
  bootstrapStatus,
  connectionState,
}: {
  logs: PrefectLogEntry[];
  status: StageRunStatus;
  bootstrapStatus: QueryStatus;
  connectionState: PrefectSocketConnectionState;
}) {
  const [open, setOpen] = useState(false);
  const isRunning = status === "running";

  const displayLogs = useMemo(
    () => (logs.length > MAX_DISPLAY_LOGS ? logs.slice(-MAX_DISPLAY_LOGS) : logs),
    [logs],
  );
  const trimmed = logs.length - displayLogs.length;

  if (logs.length === 0 && !isRunning) return null;

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

  // While running, show logs inline in the card body (no collapsible)
  if (isRunning) {
    return (
      <div>
        {connectionState === "error" && (
          <p className="mb-2 text-xs text-destructive">
            Live log stream disconnected. Prefect `logs/out` must be available for running-stage
            logs.
          </p>
        )}
        <div className="inline-flex items-center gap-1.5 text-xs text-muted-foreground mb-2">
          <Terminal className="h-3.5 w-3.5" />
          Logs
          {logs.length > 0 && (
            <span className="text-muted-foreground/50">
              ({logs.length}
              {trimmed > 0 ? `, showing last ${MAX_DISPLAY_LOGS}` : ""})
            </span>
          )}
        </div>
        <VirtualizedLogList
          logs={displayLogs}
          emptyMessage={emptyMessage}
          autoScroll
          className="bg-muted/30 p-2"
        />
      </div>
    );
  }

  // Completed/failed: collapsible toggle
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
          <span className="text-muted-foreground/50">
            ({logs.length}
            {trimmed > 0 ? `, showing last ${MAX_DISPLAY_LOGS}` : ""})
          </span>
        )}
      </button>
      {open && (
        <VirtualizedLogList
          logs={displayLogs}
          emptyMessage={emptyMessage}
          autoScroll={false}
          className="mt-2 bg-muted/30 p-2"
        />
      )}
    </div>
  );
}
