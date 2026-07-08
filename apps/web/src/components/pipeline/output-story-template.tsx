"use client";

import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { TransitionRunStatus } from "@/lib/hooks/use-run-events";
import type { LLMTrace, TransitionMeta } from "@nof1-causal-lab/api-types";
import type { ReactNode } from "react";
import { OutputPresentationShell } from "./output-presentation-shell";

export function OutputStoryLayout({
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

export type OutputStoryTemplateProps = {
  output: TransitionMeta;
  status: TransitionRunStatus;
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

export function OutputStoryTemplate({
  output,
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
}: OutputStoryTemplateProps) {
  return (
    <OutputPresentationShell
      output={output}
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
    </OutputPresentationShell>
  );
}
