"use client";

import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { LLMTrace, StageMeta } from "@nof1-causal-lab/api-types";
import { type FormEvent, type ReactNode, useState } from "react";
import { StagePresentationShellView } from "./stage-presentation-shell";

function StoryTracePanel({ trace, interactive }: { trace: LLMTrace; interactive: boolean }) {
  const [input, setInput] = useState("");

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
  }

  return (
    <LLMTracePanelView
      trace={trace}
      canRefine={interactive}
      input={input}
      onInputChange={setInput}
      onSubmit={handleSubmit}
    />
  );
}

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
  invalidated?: boolean;
  trace?: LLMTrace;
  panelContent?: ReactNode;
  interactive?: boolean;
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
  invalidated,
  trace,
  panelContent,
  interactive = stage.interactive,
  defaultPanelOpen = false,
  children,
}: StageStoryTemplateProps) {
  return (
    <StagePresentationShellView
      stage={stage}
      status={status}
      context={context}
      errorMessage={errorMessage}
      elapsedMs={elapsedMs}
      loadingHint={loadingHint}
      actions={actions}
      runningContent={runningContent}
      invalidated={invalidated}
      interactive={interactive}
      defaultPanelOpen={defaultPanelOpen}
      panelContent={
        panelContent ??
        (trace ? <StoryTracePanel trace={trace} interactive={interactive} /> : undefined)
      }
    >
      {children}
    </StagePresentationShellView>
  );
}
