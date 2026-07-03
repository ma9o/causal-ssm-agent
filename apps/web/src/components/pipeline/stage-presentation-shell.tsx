"use client";

import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { StageMeta } from "@nof1-causal-lab/api-types";
import type { ReactNode } from "react";
import { StageSection } from "./stage-section";
import { StageWithTrace, StageWithTraceView } from "./stage-with-trace";

type StagePresentationMeta = Pick<
  StageMeta,
  "id" | "number" | "label" | "description" | "loadingHint" | "interactive"
>;

export type StagePresentationShellProps = {
  stage: StagePresentationMeta;
  status: StageRunStatus;
  context?: string;
  errorMessage?: string;
  elapsedMs?: number;
  loadingHint?: string;
  actions?: ReactNode;
  runningContent?: ReactNode;
  invalidated?: boolean;
  staleArtifactIds?: string[];
  panelContent?: ReactNode;
  interactive?: boolean;
  defaultPanelOpen?: boolean;
  children?: ReactNode;
};

function renderStageSection({
  stage,
  status,
  context,
  errorMessage,
  elapsedMs,
  loadingHint,
  actions,
  runningContent,
  invalidated,
  staleArtifactIds,
  children,
}: StagePresentationShellProps) {
  return (
    <StageSection
      number={stage.number}
      title={stage.label}
      status={status}
      context={context ?? stage.description}
      errorMessage={errorMessage}
      elapsedMs={elapsedMs}
      loadingHint={loadingHint ?? stage.loadingHint}
      actions={actions}
      runningContent={runningContent}
      invalidated={invalidated}
      staleArtifactIds={staleArtifactIds}
    >
      {children}
    </StageSection>
  );
}

function CenteredStageSection({ children }: { children: ReactNode }) {
  return <div className="mx-auto w-full max-w-[1600px]">{children}</div>;
}

export function StagePresentationShellView(props: StagePresentationShellProps) {
  const { stage, panelContent, interactive = stage.interactive, defaultPanelOpen = false } = props;
  const section = renderStageSection(props);

  if (!panelContent) {
    return <CenteredStageSection>{section}</CenteredStageSection>;
  }

  return (
    <StageWithTraceView
      interactive={interactive}
      defaultOpen={defaultPanelOpen}
      panelContent={panelContent}
    >
      {section}
    </StageWithTraceView>
  );
}

export function StagePresentationShell(props: StagePresentationShellProps) {
  const { stage, panelContent, interactive = stage.interactive, defaultPanelOpen = false } = props;
  const section = renderStageSection(props);

  if (!panelContent) {
    return <CenteredStageSection>{section}</CenteredStageSection>;
  }

  return (
    <StageWithTrace
      stageId={stage.id}
      interactive={interactive}
      defaultOpen={defaultPanelOpen}
      panelContent={panelContent}
    >
      {section}
    </StageWithTrace>
  );
}
