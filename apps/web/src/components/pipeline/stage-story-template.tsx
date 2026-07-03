"use client";

import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { LLMTrace, StageMeta } from "@nof1-causal-lab/api-types";
import type { ReactNode } from "react";
import { StagePresentationShell } from "./stage-presentation-shell";

export function StageStoryLayout({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <TooltipProvider>
      <div className={cn("w-full p-4", className)}>{children}</div>
    </TooltipProvider>
  );
}

export type StageStoryTemplateProps = {
  stage: StageMeta;
  status: StageRunStatus;
  context?: string;
  errorMessage?: string;
  elapsedMs?: number;
  loadingHint?: string;
  actions?: ReactNode;
  runningContent?: ReactNode;
  trace?: LLMTrace;
  panelContent?: ReactNode;
  defaultPanelOpen?: boolean;
  children?: ReactNode;
};

export function StageStoryTemplate({
  stage,
  status,
  context,
  errorMessage,
  elapsedMs,
  loadingHint,
  actions,
  runningContent,
  trace,
  panelContent,
  defaultPanelOpen = false,
  children,
}: StageStoryTemplateProps) {
  return (
    <StagePresentationShell
      stage={stage}
      status={status}
      context={context}
      errorMessage={errorMessage}
      elapsedMs={elapsedMs}
      loadingHint={loadingHint}
      actions={actions}
      runningContent={runningContent}
      defaultPanelOpen={defaultPanelOpen}
      panelContent={panelContent ?? (trace ? <LLMTracePanelView trace={trace} /> : undefined)}
    >
      {children}
    </StagePresentationShell>
  );
}
