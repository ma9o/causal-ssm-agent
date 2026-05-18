"use client";

import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { StageMeta, StageOutcome } from "@nof1-causal-lab/api-types";
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
  outcome?: StageOutcome;
  elapsedMs?: number;
  loadingHint?: string;
  actions?: ReactNode;
  runningContent?: ReactNode;
  invalidated?: boolean;
  logView?: ReactNode;
  panelContent?: ReactNode;
  interactive?: boolean;
  defaultPanelOpen?: boolean;
  children?: ReactNode;
};

function renderStageSection({
  stage,
  status,
  context,
  outcome,
  elapsedMs,
  loadingHint,
  actions,
  runningContent,
  invalidated,
  logView,
  children,
}: StagePresentationShellProps) {
  return (
    <StageSection
      id={stage.id}
      number={stage.number}
      title={stage.label}
      status={status}
      context={context ?? stage.description}
      outcome={outcome}
      elapsedMs={elapsedMs}
      loadingHint={loadingHint ?? stage.loadingHint}
      actions={actions}
      runningContent={runningContent}
      invalidated={invalidated}
      logView={logView}
    >
      {children}
    </StageSection>
  );
}

function CenteredStageSection({ children }: { children: ReactNode }) {
  return <div className="max-w-6xl mx-auto">{children}</div>;
}

export function StagePresentationShellView(props: StagePresentationShellProps) {
  const {
    stage,
    panelContent,
    interactive = stage.interactive,
    defaultPanelOpen = false,
  } = props;
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
  const {
    stage,
    panelContent,
    interactive = stage.interactive,
    defaultPanelOpen = false,
  } = props;
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
