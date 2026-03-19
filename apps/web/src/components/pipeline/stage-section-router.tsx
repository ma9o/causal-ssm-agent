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

function StageWithTrace({
  children,
  trace,
  userId,
  stageId,
  interactive = true,
}: {
  children: ReactNode;
  trace?: LLMTrace;
  userId: string;
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
              Show LLM Trace
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
              Hide LLM Trace
            </button>
            <div className="min-h-0 flex-1 flex flex-col rounded-lg border bg-muted/30 p-3">
              <LLMTracePanel
                trace={trace}
                userId={userId}
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
  userId,
  status,
  timing,
  stageRun,
}: {
  stage: StageMeta;
  userId: string;
  status: StageRunStatus;
  timing?: StageTiming;
  stageRun?: AnalysisStageRun;
}) {
  const queryClient = useQueryClient();
  const { isInvalidated } = useRefinement();
  const invalidated = isInvalidated(stage.id);
  const isCompleted = status === "completed";
  const elapsedMs =
    timing?.completedAt && timing?.startedAt ? timing.completedAt - timing.startedAt : undefined;

  // Read context + trace + gate override + outcome from the stage data (once, after completion).
  const { data: stageData } = useStageData<{
    context?: string;
    llm_trace?: LLMTrace;
    gate_overridden?: GateOverride;
    outcome?: StageOutcome;
  }>(userId, stage.id, isCompleted);

  const outcome: StageOutcome = stageData?.outcome ?? "success";

  // Sync outcome into pipeline progress so the progress bar can reflect it
  useEffect(() => {
    if (outcome === "success") return;
    queryClient.setQueryData<PipelineProgress>(["pipeline", userId, "status"], (old) => {
      if (!old) return old;
      if (old.stageOutcomes[stage.id] === outcome) return old;
      return {
        ...old,
        stageOutcomes: { ...old.stageOutcomes, [stage.id]: outcome },
        isFailed: outcome === "fail" ? true : old.isFailed,
      };
    });
  }, [outcome, queryClient, userId, stage.id]);

  const isStage2Running = stage.id === "stage-2" && status === "running";

  const section = (
    <StageSection
      id={stage.id}
      number={stage.number}
      title={stage.label}
      status={status}
      elapsedMs={elapsedMs}
      context={stage.description}
      hasGate={stage.hasGate}
      gateOverridden={stageData?.gate_overridden}
      outcome={outcome}
      loadingHint={stage.loadingHint}
      runningContent={
        isStage2Running ? (
          <Suspense fallback={null}>
            <Stage2RunningContent
              userId={userId}
              stageStatus={status}
              stageSubflowRunId={stageRun?.stageSubflowRunId ?? null}
            />
          </Suspense>
        ) : undefined
      }
      userId={userId}
      stageSubflowRunId={stageRun?.stageSubflowRunId ?? null}
      invalidated={invalidated}
    >
      {isCompleted && (
        <ErrorBoundary>
          <Suspense fallback={null}>
            <StageContent stageId={stage.id} userId={userId} />
          </Suspense>
        </ErrorBoundary>
      )}
    </StageSection>
  );

  if (stageData?.llm_trace) {
    return (
      <StageWithTrace
        trace={stageData.llm_trace}
        userId={userId}
        stageId={stage.id}
        interactive={stage.interactive}
      >
        {section}
      </StageWithTrace>
    );
  }

  return <div className="max-w-6xl mx-auto">{section}</div>;
}

function SimpleStageWrapper<T>({
  userId,
  stageId,
  Component,
}: {
  userId: string;
  stageId: StageId;
  Component: ComponentType<{ data: T }>;
}) {
  const { data } = useStageData<T>(userId, stageId, true);
  if (!data) return null;
  return <Component data={data} />;
}

function Stage4Wrapper({ userId }: { userId: string }) {
  const { data } = useStageData<Stage4Data>(userId, "stage-4", true);
  const { data: stage2 } = useStageData<Stage2Data>(userId, "stage-2", true);
  const { data: stage1b } = useStageData<Stage1bData>(userId, "stage-1b", true);
  if (!data) return null;
  return (
    <Stage4Content
      data={data}
      extractions={stage2?.combined_extractions_sample}
      indicators={stage1b?.causal_spec.measurement.indicators}
    />
  );
}

function Stage5bWrapper({ userId }: { userId: string }) {
  const { data } = useStageData<Stage5bData>(userId, "stage-5b", true);
  if (!data) return null;
  return <Stage5bContent userId={userId} data={data} />;
}

function StageContent({ stageId, userId }: { stageId: string; userId: string }) {
  switch (stageId) {
    case "stage-0":
      return (
        <SimpleStageWrapper<Stage0Data>
          userId={userId}
          stageId="stage-0"
          Component={Stage0Content}
        />
      );
    case "stage-1a":
      return (
        <SimpleStageWrapper<Stage1aData>
          userId={userId}
          stageId="stage-1a"
          Component={Stage1aContent}
        />
      );
    case "stage-1b":
      return (
        <SimpleStageWrapper<Stage1bData>
          userId={userId}
          stageId="stage-1b"
          Component={Stage1bContent}
        />
      );
    case "stage-2":
      return (
        <SimpleStageWrapper<Stage2Data>
          userId={userId}
          stageId="stage-2"
          Component={Stage2Content}
        />
      );
    case "stage-3":
      return (
        <SimpleStageWrapper<Stage3Data>
          userId={userId}
          stageId="stage-3"
          Component={Stage3Content}
        />
      );
    case "stage-4":
      return <Stage4Wrapper userId={userId} />;
    case "stage-4b":
      return (
        <SimpleStageWrapper<Stage4bData>
          userId={userId}
          stageId="stage-4b"
          Component={Stage4bContent}
        />
      );
    case "stage-5a":
      return (
        <SimpleStageWrapper<Stage5aData>
          userId={userId}
          stageId="stage-5a"
          Component={Stage5aContent}
        />
      );
    case "stage-5b":
      return <Stage5bWrapper userId={userId} />;
    case "stage-6":
      return (
        <SimpleStageWrapper<Stage6Data>
          userId={userId}
          stageId="stage-6"
          Component={Stage6Content}
        />
      );
    default:
      return null;
  }
}
