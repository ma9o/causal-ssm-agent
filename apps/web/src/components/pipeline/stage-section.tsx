"use client";

import { Skeleton } from "@/components/ui/skeleton";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { AlertCircle, ChevronDown, RotateCcw } from "lucide-react";
import { motion } from "motion/react";
import prettyMs from "pretty-ms";
import { type ReactNode, useState } from "react";
import { StageHeader } from "./stage-header";

export function StageSection({
  id,
  number,
  title,
  status,
  context,
  children,
  defaultCollapsed = false,
  elapsedMs,
  errorMessage,
  loadingHint,
  runningContent,
  invalidated = false,
  actions,
}: {
  id?: string;
  number: string;
  title: string;
  status: StageRunStatus;
  context?: string;
  children?: ReactNode;
  defaultCollapsed?: boolean;
  elapsedMs?: number;
  /** Failure detail shown when the stage run raised. */
  errorMessage?: string;
  loadingHint?: string;
  runningContent?: ReactNode;
  invalidated?: boolean;
  /** Optional actions rendered top-right of the card header. */
  actions?: ReactNode;
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

  const isCollapsible = status === "completed" && !invalidated;

  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: invalidated ? 0.45 : 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={`scroll-mt-28 rounded-lg border bg-card p-4 shadow-sm sm:p-6 ${invalidated ? "pointer-events-none border-dashed border-muted-foreground/30" : ""}`}
    >
      <div
        className={`flex items-start gap-3${isCollapsible ? " cursor-pointer" : ""}`}
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
          <StageHeader number={number} title={title} status={status} context={context} />
          {invalidated && (
            <span className="mt-1 inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 text-xs text-muted-foreground">
              <RotateCcw className="h-3 w-3" />
              Needs re-run
            </span>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2 pt-1">
          {actions}
          {isCollapsible && (
            <>
              {elapsedMs !== undefined && (
                <span className="text-xs text-muted-foreground/60 font-mono">
                  {prettyMs(elapsedMs)}
                </span>
              )}
              <ChevronDown
                className={`h-5 w-5 shrink-0 text-muted-foreground transition-transform duration-200 ${collapsed ? "-rotate-90" : ""}`}
              />
            </>
          )}
        </div>
      </div>
      {status === "running" && (
        <motion.div
          className="mt-4 space-y-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          {runningContent ?? (
            <>
              {loadingHint && <p className="text-sm text-muted-foreground">{loadingHint}</p>}
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-32 w-full" />
            </>
          )}
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
              {errorMessage ?? "This may be a transient error."}
            </p>
          </div>
        </motion.div>
      )}
    </motion.section>
  );
}
