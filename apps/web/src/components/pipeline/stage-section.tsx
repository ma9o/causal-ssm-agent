"use client";

import { Skeleton } from "@/components/ui/skeleton";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { GateOverride, StageId, StageOutcome } from "@causal-ssm/api-types";
import { AlertCircle, ChevronDown } from "lucide-react";
import { motion } from "motion/react";
import prettyMs from "pretty-ms";
import { type ReactNode, useState } from "react";
import { StageHeader } from "./stage-header";
import { StageLogViewer } from "./stage-log-viewer";

export function StageSection({
  id,
  number,
  title,
  status,
  context,
  children,
  defaultCollapsed = false,
  elapsedMs,
  hasGate = false,
  gateOverridden,
  outcome = "success",
  loadingHint,
  runId,
  stageId,
}: {
  id?: string;
  number: string;
  title: string;
  status: StageRunStatus;
  context?: string;
  children?: ReactNode;
  defaultCollapsed?: boolean;
  elapsedMs?: number;
  hasGate?: boolean;
  gateOverridden?: GateOverride;
  outcome?: StageOutcome;
  loadingHint?: string;
  runId?: string;
  stageId?: StageId;
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [prevStatus, setPrevStatus] = useState(status);

  // Expand when transitioning to completed (derive-state-from-props)
  if (status !== prevStatus) {
    setPrevStatus(status);
    if (status === "completed") {
      setCollapsed(false);
    }
  }

  // Failed-outcome stages should not be collapsible — the failure must remain visible
  const isCollapsible = status === "completed" && outcome !== "fail";

  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="scroll-mt-28 rounded-lg border bg-card p-4 shadow-sm sm:p-6"
    >
      <div
        className={isCollapsible ? "flex items-start gap-3 cursor-pointer" : ""}
        role={isCollapsible ? "button" : undefined}
        tabIndex={isCollapsible ? 0 : undefined}
        aria-expanded={isCollapsible ? !collapsed : undefined}
        onClick={isCollapsible ? () => setCollapsed((c) => !c) : undefined}
        onKeyDown={
          isCollapsible
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setCollapsed((c) => !c);
                }
              }
            : undefined
        }
      >
        <div className="flex-1 min-w-0">
          <StageHeader
            number={number}
            title={title}
            status={status}
            hasGate={hasGate}
            gateOverridden={gateOverridden}
            outcome={outcome}
            context={context}
          />
        </div>
        {isCollapsible && (
          <div className="flex shrink-0 items-center gap-2 pt-1">
            {elapsedMs !== undefined && (
              <span className="text-xs text-muted-foreground/60 font-mono">
                {prettyMs(elapsedMs)}
              </span>
            )}
            <ChevronDown
              className={`h-5 w-5 shrink-0 text-muted-foreground transition-transform duration-200 ${collapsed ? "-rotate-90" : ""}`}
            />
          </div>
        )}
      </div>
      {status === "running" && (
        <motion.div
          className="mt-4 space-y-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          {loadingHint && <p className="text-sm text-muted-foreground">{loadingHint}</p>}
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-32 w-full" />
        </motion.div>
      )}
      {status === "completed" && !collapsed && (
        <motion.div
          className="mt-4 space-y-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          {children}
        </motion.div>
      )}
      {status === "failed" && (
        <motion.div
          className="mt-4 flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
          <div className="text-sm">
            <p className="font-medium text-destructive">Stage failed</p>
            <p className="mt-0.5 text-muted-foreground">
              Check pipeline logs for details. This may be a transient error.
            </p>
          </div>
        </motion.div>
      )}
      {runId && stageId && status !== "pending" && (
        <StageLogViewer runId={runId} stageId={stageId} status={status} />
      )}
    </motion.section>
  );
}
