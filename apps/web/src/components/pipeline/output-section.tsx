"use client";

import { Skeleton } from "@/components/ui/skeleton";
import type { TransitionRunStatus } from "@/lib/hooks/use-run-events";
import { AlertCircle, ChevronDown } from "lucide-react";
import { motion } from "motion/react";
import prettyMs from "pretty-ms";
import { type ReactNode, useState } from "react";
import { OutputHeader } from "./output-header";

export function OutputSection({
  id,
  title,
  status,
  context,
  children,
  defaultCollapsed = false,
  elapsedMs,
  errorMessage,
  loadingHint,
  runningContent,
  actions,
  staleArtifactIds,
}: {
  id?: string;
  title: string;
  status: TransitionRunStatus;
  context?: string;
  children?: ReactNode;
  defaultCollapsed?: boolean;
  elapsedMs?: number;
  /** Failure detail shown when the transition run raised. */
  errorMessage?: string;
  loadingHint?: string;
  runningContent?: ReactNode;
  /** Optional actions rendered top-right of the card header. */
  actions?: ReactNode;
  /** Stale artifacts this transition produced (backend freshness report). */
  staleArtifactIds?: string[];
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

  const isCollapsible = status === "completed";

  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="scroll-mt-28 rounded-lg border bg-card p-4 shadow-sm sm:p-6"
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
          <OutputHeader
            title={title}
            status={status}
            context={context}
            staleArtifactIds={staleArtifactIds}
          />
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
          className="mt-4 space-y-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 p-3">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
            <div className="text-sm">
              <p className="font-medium text-destructive">Transition failed</p>
              <p className="mt-0.5 text-muted-foreground">
                {errorMessage ?? "This may be a transient error."}
              </p>
            </div>
          </div>
          {runningContent}
        </motion.div>
      )}
    </motion.section>
  );
}
