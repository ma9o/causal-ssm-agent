"use client";

import { LLMTracePanel } from "@/components/ui/custom/llm-trace-panel";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import type { AnalysisStageRun } from "@/lib/api/analysis";
import { useRefinement } from "@/lib/contexts/refinement-context";
import type { PipelineProgress, StageRunStatus, StageTiming } from "@/lib/hooks/use-run-events";
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
import { Wrench } from "lucide-react";
import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
} from "react";
import { StageSection } from "./stage-section";
import { StageWithTrace } from "./stage-with-trace";

const Stage0Content = lazy(() => import("./stage-contents/stage-0-content"));
const Stage1aContent = lazy(() => import("./stage-contents/stage-1a-content"));
const Stage1bContent = lazy(() => import("./stage-contents/stage-1b-content"));
const Stage2Content = lazy(() => import("./stage-contents/stage-2-content"));
const Stage2RunningContent = lazy(() => import("./stage-contents/stage-2-running-content"));
const Stage3Content = lazy(() => import("./stage-contents/stage-3-content"));
import { buildFixPrompt } from "./stage-contents/stage-3-content";
const Stage4Content = lazy(() => import("./stage-contents/stage-4-content"));
const Stage4bContent = lazy(() => import("./stage-contents/stage-4b-content"));
const Stage5aContent = lazy(() => import("./stage-contents/stage-5a-content"));
const Stage5bContent = lazy(() => import("./stage-contents/stage-5b-content"));
const Stage6Content = lazy(() => import("./stage-contents/stage-6-content"));

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

export function StageSectionRouter({
  stage,
  workspaceId,
  status,
  timing,
  stageRun,
}: {
  stage: StageMeta;
  workspaceId: string;
  status: StageRunStatus;
  timing?: StageTiming;
  stageRun?: AnalysisStageRun;
}) {
  const queryClient = useQueryClient();
  const { isInvalidated, pendingStagePatches, refiningStageId, setPrefill } = useRefinement();
  const invalidated = isInvalidated(stage.id);
  const isCompleted = status === "completed";
  const elapsedMs =
    timing?.completedAt && timing?.startedAt ? timing.completedAt - timing.startedAt : undefined;

  // Read context + trace + outcome from the stage data (once, after completion).
  const { data: stageData } = useStageData<StageViewData>(workspaceId, stage.id, isCompleted);
  const pendingStagePatch =
    refiningStageId === stage.id ? pendingStagePatches[stage.id] ?? null : null;
  const projectedStageData =
    stageData && pendingStagePatch ? ({ ...stageData, ...pendingStagePatch } as StageViewData) : stageData;

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

  const isStage2Running = stage.id === "stage-2" && status === "running";

  const handleFixMeasurements = useCallback(() => {
    if (!projectedStageData || stage.id !== "stage-3") return;
    const prompt = buildFixPrompt(projectedStageData as Stage3Data);
    if (!prompt) return;
    setPrefill("stage-1b", prompt);
    requestAnimationFrame(() => {
      document.getElementById("stage-1b")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }, [projectedStageData, stage.id, setPrefill]);

  const showFixButton =
    stage.id === "stage-3" && isCompleted && (outcome === "fail" || outcome === "warn");

  const section = (
    <StageSection
      id={stage.id}
      stageId={stage.id}
      number={stage.number}
      title={stage.label}
      status={status}
      elapsedMs={elapsedMs}
      context={stage.description}
      outcome={outcome}
      loadingHint={stage.loadingHint}
      actions={
        showFixButton ? (
          <button
            type="button"
            onClick={handleFixMeasurements}
            className="inline-flex items-center gap-1.5 rounded-md border border-warning/50 bg-warning/10 px-3 py-1.5 text-xs font-medium text-warning-foreground transition-colors hover:bg-warning/20"
          >
            <Wrench className="h-3.5 w-3.5" />
            Fix measurements
          </button>
        ) : undefined
      }
      runningContent={
        isStage2Running ? (
          <Suspense fallback={null}>
            <Stage2RunningContent
              workspaceId={workspaceId}
              rootFlowRunId={stageRun?.ownerRootFlowRunId ?? null}
              stageStatus={status}
              logFlowRunIds={stageRun?.logFlowRunIds ?? []}
            />
          </Suspense>
        ) : undefined
      }
      workspaceId={workspaceId}
      logFlowRunIds={stageRun?.logFlowRunIds ?? []}
      invalidated={invalidated}
      showLogViewer={!isStage2Running}
    >
      {isCompleted && (
        <ErrorBoundary>
          <Suspense fallback={null}>
            <StageContent stageId={stage.id} workspaceId={workspaceId} data={projectedStageData} />
          </Suspense>
        </ErrorBoundary>
      )}
    </StageSection>
  );

  if (projectedStageData?.llm_trace) {
    return (
      <StageWithTrace
        stageId={stage.id}
        interactive={stage.interactive}
        panelContent={
          <LLMTracePanel
            trace={projectedStageData.llm_trace}
            workspaceId={workspaceId}
            stageId={stage.id}
            interactive={stage.interactive}
          />
        }
      >
        {section}
      </StageWithTrace>
    );
  }

  return <div className="max-w-6xl mx-auto">{section}</div>;
}

function Stage4Wrapper({ workspaceId, data }: { workspaceId: string; data: Stage4Data }) {
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

function Stage5bWrapper({ workspaceId, data }: { workspaceId: string; data: Stage5bData }) {
  return <Stage5bContent workspaceId={workspaceId} data={data} />;
}

function Stage6Wrapper({ data }: { data: Stage6Data }) {
  return <Stage6Content data={data} />;
}

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

  switch (stageId) {
    case "stage-0":
      return <Stage0Content data={data as Stage0Data} />;
    case "stage-1a":
      return <Stage1aContent data={data as Stage1aData} />;
    case "stage-1b":
      return <Stage1bContent data={data as Stage1bData} />;
    case "stage-2":
      return <Stage2Content data={data as Stage2Data} />;
    case "stage-3":
      return <Stage3Content data={data as Stage3Data} />;
    case "stage-4":
      return <Stage4Wrapper workspaceId={workspaceId} data={data as Stage4Data} />;
    case "stage-4b":
      return <Stage4bContent data={data as Stage4bData} />;
    case "stage-5a":
      return <Stage5aContent data={data as Stage5aData} />;
    case "stage-5b":
      return <Stage5bWrapper workspaceId={workspaceId} data={data as Stage5bData} />;
    case "stage-6":
      return <Stage6Wrapper data={data as Stage6Data} />;
    default:
      return null;
  }
}
