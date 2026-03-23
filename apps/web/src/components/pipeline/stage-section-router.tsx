"use client";

import { LLMTracePanel } from "@/components/ui/custom/llm-trace-panel";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import type { AnalysisStageRun } from "@/lib/api/analysis";
import { useRefinement } from "@/lib/contexts/refinement-context";
import type { PipelineProgress, StageRunStatus, StageTiming } from "@/lib/hooks/use-run-events";
import { useStageData } from "@/lib/hooks/use-stage-data";
import { cn } from "@/lib/utils";
import type {
  GateOverride,
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
  StageId,
  StageMeta,
  StageOutcome,
} from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { Bot } from "lucide-react";
import { motion } from "motion/react";
import {
  type ComponentType,
  type ReactNode,
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { StageSection } from "./stage-section";

const Stage0Content = lazy(() => import("./stage-contents/stage-0-content"));
const Stage1aContent = lazy(() => import("./stage-contents/stage-1a-content"));
const Stage1bContent = lazy(() => import("./stage-contents/stage-1b-content"));
const Stage2Content = lazy(() => import("./stage-contents/stage-2-content"));
const Stage2RunningContent = lazy(() => import("./stage-contents/stage-2-running-content"));
const Stage3Content = lazy(() => import("./stage-contents/stage-3-content"));
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
  gate_overridden?: GateOverride;
  outcome?: StageOutcome;
};

function StageWithTrace({
  children,
  trace,
  workspaceId,
  stageId,
  interactive = true,
}: {
  children: ReactNode;
  trace?: LLMTrace;
  workspaceId: string;
  stageId: string;
  interactive?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const leftRef = useRef<HTMLDivElement>(null);
  const [leftHeight, setLeftHeight] = useState<number | undefined>(undefined);

  const measureLeft = useCallback(() => {
    if (leftRef.current) setLeftHeight(leftRef.current.offsetHeight);
  }, []);

  useEffect(() => {
    if (!isOpen || !leftRef.current) return;
    measureLeft();
    const ro = new ResizeObserver(measureLeft);
    ro.observe(leftRef.current);
    return () => ro.disconnect();
  }, [isOpen, measureLeft]);

  if (!trace) return <>{children}</>;

  const transition = { duration: 0.35, ease: [0.4, 0, 0.2, 1] as const };

  return (
    <div className={cn("flex", isOpen && "items-start gap-4")}>
      <motion.div
        ref={leftRef}
        className={cn("min-w-0", !isOpen && "max-w-6xl mx-auto w-full")}
        animate={{ flex: isOpen ? 2 : 1 }}
        transition={transition}
      >
        {!isOpen && (
          <div className="mb-2 flex justify-end">
            <button
              type="button"
              onClick={() => setIsOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-md border border-muted bg-muted/50 px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted"
            >
              <Bot className="h-3.5 w-3.5" />
              Show Assistant Details
            </button>
          </div>
        )}
        {children}
      </motion.div>
      <motion.div
        className={cn("min-w-0", !isOpen && "h-0 overflow-hidden")}
        style={isOpen && leftHeight ? { height: leftHeight } : undefined}
        animate={{ flex: isOpen ? 1 : 0, opacity: isOpen ? 1 : 0 }}
        initial={false}
        transition={transition}
      >
        {isOpen && (
          <div className="flex h-full flex-col gap-3">
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="inline-flex w-full shrink-0 items-center justify-center gap-1.5 rounded-md border border-primary/30 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary transition-colors"
            >
              <Bot className="h-3.5 w-3.5" />
              Hide Assistant Details
            </button>
            <div className="min-h-0 flex-1 flex flex-col rounded-lg border bg-muted/30 p-3">
              <LLMTracePanel
                trace={trace}
                workspaceId={workspaceId}
                stageId={stageId}
                interactive={interactive}
              />
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}

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
  const { isInvalidated, pendingStagePatches, refiningStageId } = useRefinement();
  const invalidated = isInvalidated(stage.id);
  const isCompleted = status === "completed";
  const elapsedMs =
    timing?.completedAt && timing?.startedAt ? timing.completedAt - timing.startedAt : undefined;

  // Read context + trace + gate override + outcome from the stage data (once, after completion).
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
        isFailed: outcome === "fail" ? true : old.isFailed,
      };
    });
  }, [outcome, queryClient, workspaceId, stage.id]);

  const isStage2Running = stage.id === "stage-2" && status === "running";

  const section = (
    <StageSection
      id={stage.id}
      stageId={stage.id}
      number={stage.number}
      title={stage.label}
      status={status}
      elapsedMs={elapsedMs}
      context={stage.description}
      hasGate={stage.hasGate}
      gateOverridden={projectedStageData?.gate_overridden}
      outcome={outcome}
      loadingHint={stage.loadingHint}
      runningContent={
        isStage2Running ? (
          <Suspense fallback={null}>
            <Stage2RunningContent
              workspaceId={workspaceId}
              rootFlowRunId={stageRun?.ownerRootFlowRunId ?? null}
              stageStatus={status}
              stageSubflowRunId={stageRun?.stageSubflowRunId ?? null}
              logFlowRunIds={stageRun?.logFlowRunIds ?? []}
            />
          </Suspense>
        ) : undefined
      }
      workspaceId={workspaceId}
      stageSubflowRunId={stageRun?.stageSubflowRunId ?? null}
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
        trace={projectedStageData.llm_trace}
        workspaceId={workspaceId}
        stageId={stage.id}
        interactive={stage.interactive}
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
