"use client";

import { LLMTracePanel } from "@/components/ui/custom/llm-trace-panel";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import type { AnalysisStageRun } from "@/lib/api/analysis";
import { useRefinement } from "@/lib/contexts/refinement-context";
import type { PipelineProgress, StageRunStatus, StageTiming } from "@/lib/hooks/use-run-events";
import { useStageLogs } from "@/lib/hooks/use-stage-logs";
import { useStageData } from "@/lib/hooks/use-stage-data";
import type {
  LLMTrace,
  Stage0Data,
  Stage1aData,
  Stage1bData,
  Stage2Data,
  Stage3Data,
  Stage4Data,
  Stage4bData,
  Stage5aData,
  Stage5bData,
  Stage6Data,
  StageMeta,
  StageOutcome,
} from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import {
  Suspense,
  lazy,
  memo,
  type ElementType,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
} from "react";
import { StageLogView } from "./stage-log-viewer";
import { StagePresentationShell } from "./stage-presentation-shell";
import { Stage3FixAction } from "./stage-contents/stage-3-content";
import { buildStage6DagScene } from "./stage-contents/stage-6-presentation";

const Stage0Content = lazy(() => import("./stage-contents/stage-0-content"));
const Stage1aContent = lazy(() => import("./stage-contents/stage-1a-content"));
const Stage1bContent = lazy(() => import("./stage-contents/stage-1b-content"));
const Stage2Content = lazy(() => import("./stage-contents/stage-2-content"));
const Stage2RunningContent = lazy(() => import("./stage-contents/stage-2-running-content"));
const Stage4RunningContent = lazy(() => import("./stage-contents/stage-4-running-content"));
const Stage3Content = lazy(() => import("./stage-contents/stage-3-content"));
const Stage4Content = lazy(() => import("./stage-contents/stage-4-content"));
const Stage4bContent = lazy(() => import("./stage-contents/stage-4b-content"));
const Stage5aContent = lazy(() => import("./stage-contents/stage-5a-content"));
const Stage5bContent = lazy(() => import("./stage-contents/stage-5b-content"));
const Stage6Showcase = lazy(() => import("./stage-contents/stage-6-showcase"));

type AnyStageData =
  | Stage0Data
  | Stage1aData
  | Stage1bData
  | Stage2Data
  | Stage3Data
  | Stage4Data
  | Stage4bData
  | Stage5aData
  | Stage5bData
  | Stage6Data;

type StageViewData = AnyStageData & {
  context?: string;
  llm_trace?: LLMTrace;
  outcome?: StageOutcome;
};

type StageSectionRouterProps = {
  stage: StageMeta;
  workspaceId: string;
  status: StageRunStatus;
  timing?: StageTiming;
  stageRun?: AnalysisStageRun;
};

function stageRunsEqual(previous?: AnalysisStageRun, next?: AnalysisStageRun): boolean {
  return (
    previous?.ownerRootFlowRunId === next?.ownerRootFlowRunId &&
    previous?.stageSubflowRunId === next?.stageSubflowRunId &&
    previous?.execution?.stateType === next?.execution?.stateType &&
    previous?.execution?.startTime === next?.execution?.startTime &&
    previous?.execution?.endTime === next?.execution?.endTime &&
    (previous?.initialLogFlowRunIds.join("|") ?? "") ===
      (next?.initialLogFlowRunIds.join("|") ?? "")
  );
}

function StageSectionRouterInner({
  stage,
  workspaceId,
  status,
  timing,
  stageRun,
}: StageSectionRouterProps) {
  const queryClient = useQueryClient();
  const { isInvalidated, pendingStagePatches, refiningStageId, setPrefill } = useRefinement();
  const invalidated = isInvalidated(stage.id);
  const isCompleted = status === "completed";
  const elapsedMs =
    timing?.completedAt && timing?.startedAt ? timing.completedAt - timing.startedAt : undefined;

  // Read context + trace + outcome from the stage data (once, after completion).
  const { data: stageData } = useStageData<StageViewData>(workspaceId, stage.id, isCompleted);
  const pendingStagePatch =
    refiningStageId === stage.id ? (pendingStagePatches[stage.id] ?? null) : null;
  const projectedStageData = useMemo(
    () =>
      stageData && pendingStagePatch
        ? ({ ...stageData, ...pendingStagePatch } as StageViewData)
        : stageData,
    [pendingStagePatch, stageData],
  );

  const outcome: StageOutcome = projectedStageData?.outcome ?? "success";

  // Sync outcome into pipeline progress so the progress bar can reflect it
  useEffect(() => {
    if (outcome === "success") return;
    queryClient.setQueryData<PipelineProgress>(["pipeline", workspaceId, "status"], (old) => {
      if (!old) return old;
      if (old.stageOutcomes[stage.id] === outcome) return old;
      return {
        ...old,
        stageOutcomes: { ...old.stageOutcomes, [stage.id]: outcome },
      };
    });
  }, [outcome, queryClient, workspaceId, stage.id]);

  // Hook lives here (always mounted) so transition tracking works across
  // running→completed without remounting.
  const { logs, bootstrapStatus, connectionState } = useStageLogs(
    workspaceId,
    stage.id,
    stageRun,
    status,
  );
  const showLogViewer = status !== "pending";
  const logView = showLogViewer ? (
    <StageLogView
      logs={logs}
      status={status}
      bootstrapStatus={bootstrapStatus}
      connectionState={connectionState}
    />
  ) : undefined;

  const handleFixMeasurements = useCallback(
    (prompt: string) => {
      setPrefill("stage-1b", prompt);
      requestAnimationFrame(() => {
        document.getElementById("stage-1b")?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    },
    [setPrefill],
  );

  return (
    <StagePresentationShell
      stage={stage}
      status={status}
      elapsedMs={elapsedMs}
      context={stage.description}
      outcome={outcome}
      loadingHint={stage.loadingHint}
      actions={
        stage.id === "stage-3" && isCompleted && projectedStageData ? (
          <Stage3FixAction data={projectedStageData as Stage3Data} onFix={handleFixMeasurements} />
        ) : undefined
      }
      runningContent={
        stage.id === "stage-2" && status === "running" ? (
          <Suspense fallback={null}>
            <Stage2RunningContent
              workspaceId={workspaceId}
              stageStatus={status}
              stageRun={stageRun}
            />
          </Suspense>
        ) : stage.id === "stage-4" && status === "running" ? (
          <Suspense fallback={null}>
            <Stage4RunningContent
              workspaceId={workspaceId}
              stageStatus={status}
              stageRun={stageRun}
            />
          </Suspense>
        ) : undefined
      }
      invalidated={invalidated}
      logView={logView}
      panelContent={
        projectedStageData?.llm_trace ? (
          <LLMTracePanel
            trace={projectedStageData.llm_trace}
            workspaceId={workspaceId}
            stageId={stage.id}
            interactive={stage.interactive}
          />
        ) : undefined
      }
    >
      {isCompleted && (
        <ErrorBoundary>
          <Suspense fallback={null}>
            <StageContent stageId={stage.id} workspaceId={workspaceId} data={projectedStageData} />
          </Suspense>
        </ErrorBoundary>
      )}
    </StagePresentationShell>
  );
}

export const StageSectionRouter = memo(
  StageSectionRouterInner,
  (previous, next) =>
    previous.workspaceId === next.workspaceId &&
    previous.stage.id === next.stage.id &&
    previous.status === next.status &&
    previous.timing?.startedAt === next.timing?.startedAt &&
    previous.timing?.completedAt === next.timing?.completedAt &&
    stageRunsEqual(previous.stageRun, next.stageRun),
);

type StageContentAdapterProps = {
  workspaceId: string;
  data: AnyStageData;
};

function createStageDataAdapter<TData>(Component: ElementType<{ data: TData }>) {
  return function StageDataAdapter({ data }: StageContentAdapterProps) {
    return <Component data={data as TData} />;
  };
}

function Stage4ConnectedContent({ workspaceId, data }: { workspaceId: string; data: Stage4Data }) {
  const { data: stage2 } = useStageData<Stage2Data>(workspaceId, "stage-2", true);
  const { data: stage1b } = useStageData<Stage1bData>(workspaceId, "stage-1b", true);
  return (
    <Stage4Content
      data={data}
      extractions={stage2?.combined_extractions_sample}
      indicators={stage1b?.causal_spec.measurement.indicators}
    />
  );
}

function Stage5bConnectedContent({
  workspaceId,
  data,
}: {
  workspaceId: string;
  data: Stage5bData;
}) {
  return <Stage5bContent workspaceId={workspaceId} data={data} />;
}

function Stage6ConnectedContent({ workspaceId, data }: { workspaceId: string; data: Stage6Data }) {
  const { refinementMessages } = useRefinement();
  const { data: stage1a } = useStageData<Stage1aData>(workspaceId, "stage-1a", true);
  const { data: stage1b } = useStageData<Stage1bData>(workspaceId, "stage-1b", true);
  const { data: stage4 } = useStageData<Stage4Data>(workspaceId, "stage-4", true);
  const { data: stage5a } = useStageData<Stage5aData>(workspaceId, "stage-5a", true);
  const { data: stage5b } = useStageData<Stage5bData>(workspaceId, "stage-5b", true);
  const dagScene = useMemo(
    () =>
      buildStage6DagScene({
        stage1a,
        stage1b,
        stage4,
        stage5a,
        stage5b,
        refinementMessages: refinementMessages["stage-6"] ?? [],
        height: "600px",
      }),
    [refinementMessages, stage1a, stage1b, stage4, stage5a, stage5b],
  );

  return <Stage6Showcase data={data} dagScene={dagScene} />;
}

const stageContentAdapters = {
  "stage-0": ({ workspaceId, data }: StageContentAdapterProps) => (
    <Stage0Content workspaceId={workspaceId} data={data as Stage0Data} />
  ),
  "stage-1a": createStageDataAdapter<Stage1aData>(Stage1aContent),
  "stage-1b": createStageDataAdapter<Stage1bData>(Stage1bContent),
  "stage-2": ({ workspaceId, data }: StageContentAdapterProps) => (
    <Stage2Content workspaceId={workspaceId} data={data as Stage2Data} />
  ),
  "stage-3": createStageDataAdapter<Stage3Data>(Stage3Content),
  "stage-4": ({ workspaceId, data }: StageContentAdapterProps) => (
    <Stage4ConnectedContent workspaceId={workspaceId} data={data as Stage4Data} />
  ),
  "stage-4b": createStageDataAdapter<Stage4bData>(Stage4bContent),
  "stage-5a": createStageDataAdapter<Stage5aData>(Stage5aContent),
  "stage-5b": ({ workspaceId, data }: StageContentAdapterProps) => (
    <Stage5bConnectedContent workspaceId={workspaceId} data={data as Stage5bData} />
  ),
  "stage-6": ({ workspaceId, data }: StageContentAdapterProps) => (
    <Stage6ConnectedContent workspaceId={workspaceId} data={data as Stage6Data} />
  ),
} satisfies Record<string, (props: StageContentAdapterProps) => ReactNode>;

function StageContent({
  stageId,
  workspaceId,
  data,
}: {
  stageId: string;
  workspaceId: string;
  data?: StageViewData;
}) {
  if (!data) return null;
  const renderStageContent = stageContentAdapters[stageId as keyof typeof stageContentAdapters];
  return renderStageContent ? renderStageContent({ workspaceId, data }) : null;
}
