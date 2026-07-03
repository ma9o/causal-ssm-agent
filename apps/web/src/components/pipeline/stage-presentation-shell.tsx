"use client";

import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { StageMeta } from "@nof1-causal-lab/api-types";
import type { ReactNode } from "react";
import { StageSection } from "./stage-section";
import { StageWithTraceView } from "./stage-with-trace";

type StagePresentationMeta = Pick<
  StageMeta,
  "id" | "number" | "label" | "description" | "loadingHint"
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
  staleArtifactIds?: string[];
  panelContent?: ReactNode;
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
      staleArtifactIds={staleArtifactIds}
    >
      {children}
    </StageSection>
  );
}

function CenteredStageSection({ children }: { children: ReactNode }) {
  return <div className="mx-auto w-full max-w-[1600px]">{children}</div>;
}

export function StagePresentationShell(props: StagePresentationShellProps) {
  const { panelContent, defaultPanelOpen = false } = props;
  const section = renderStageSection(props);

  if (!panelContent) {
    return <CenteredStageSection>{section}</CenteredStageSection>;
  }

  return (
    <StageWithTraceView defaultOpen={defaultPanelOpen} panelContent={panelContent}>
      {section}
    </StageWithTraceView>
  );
}
