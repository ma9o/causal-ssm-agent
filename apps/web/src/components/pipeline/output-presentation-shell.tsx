"use client";

import type { TransitionRunStatus } from "@/lib/hooks/use-run-events";
import type { TransitionMeta } from "@nof1-causal-lab/api-types";
import type { ReactNode } from "react";
import { OutputSection } from "./output-section";
import { OutputWithTraceView } from "./output-with-trace";

type OutputPresentationMeta = Pick<TransitionMeta, "id" | "label" | "description" | "loadingHint">;

export type OutputPresentationShellProps = {
  output: OutputPresentationMeta;
  status: TransitionRunStatus;
  context?: string;
  errorMessage?: string;
  elapsedMs?: number;
  loadingHint?: string;
  actions?: ReactNode;
  runningContent?: ReactNode;
  staleArtifactIds?: string[];
  panelContent?: ReactNode;
  defaultPanelOpen?: boolean;
  children?: ReactNode;
};

function renderOutputSection({
  output,
  status,
  context,
  errorMessage,
  elapsedMs,
  loadingHint,
  actions,
  runningContent,
  staleArtifactIds,
  children,
}: OutputPresentationShellProps) {
  return (
    <OutputSection
      title={output.label}
      status={status}
      context={context ?? output.description}
      errorMessage={errorMessage}
      elapsedMs={elapsedMs}
      loadingHint={loadingHint ?? output.loadingHint}
      actions={actions}
      runningContent={runningContent}
      staleArtifactIds={staleArtifactIds}
    >
      {children}
    </OutputSection>
  );
}

function CenteredOutputSection({ children }: { children: ReactNode }) {
  return <div className="mx-auto w-full max-w-[1600px]">{children}</div>;
}

export function OutputPresentationShell(props: OutputPresentationShellProps) {
  const { panelContent, defaultPanelOpen = false } = props;
  const section = renderOutputSection(props);

  if (!panelContent) {
    return <CenteredOutputSection>{section}</CenteredOutputSection>;
  }

  return (
    <OutputWithTraceView defaultOpen={defaultPanelOpen} panelContent={panelContent}>
      {section}
    </OutputWithTraceView>
  );
}
