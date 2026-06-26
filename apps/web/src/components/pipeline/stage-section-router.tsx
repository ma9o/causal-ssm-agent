"use client";

import type {
  LLMTrace,
  Stage0Data,
  Stage1aData,
  Stage1bData,
  Stage2Data,
  Stage3Data,
  Stage4Data,
  Stage5bData,
  Stage6Data,
  StageMeta,
  StageOutcome,
} from "@nof1-causal-lab/api-types";
import { useQueryClient } from "@tanstack/react-query";
import {
  type ComponentType,
  lazy,
  memo,
  type ReactNode,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
} from "react";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import type { AnalysisStageRun } from "@/lib/api/analysis";
import { createSimulateDispatch } from "@/components/dag/interactive/dispatch-simulate";
import {
  buildDevMockMessages,
  makeMockSimulate,
  synthesizeMockScenarios,
} from "@/components/dag/interactive/dev-mock-scenario";
import { useRefinement } from "@/lib/contexts/refinement-context";
import { isSharedWorkspaceId } from "@/lib/shared-workspaces";
import type { PipelineProgress, StageRunStatus, StageTiming } from "@/lib/hooks/use-run-events";
import { useStageData } from "@/lib/hooks/use-stage-data";
import { useStageLogs } from "@/lib/hooks/use-stage-logs";
import { resolveStageObservedStatus } from "@/lib/stage-runtime";
import { Stage3FixAction } from "./stage-contents/stage-3-content";
import { buildEdgePosteriors, buildStage6Scenarios } from "./stage-contents/stage-6-scenarios";
import { StageLogView } from "./stage-log-viewer";
import { StagePresentationShell } from "./stage-presentation-shell";

const Stage0Content = lazy(() => import("./stage-contents/stage-0-content"));
const Stage1aContent = lazy(() => import("./stage-contents/stage-1a-content"));
const Stage1bContent = lazy(() => import("./stage-contents/stage-1b-content"));
const Stage2Content = lazy(() => import("./stage-contents/stage-2-content"));
const Stage2RunningContent = lazy(() => import("./stage-contents/stage-2-running-content"));
const Stage4RunningContent = lazy(() => import("./stage-contents/stage-4-running-content"));
const Stage3Content = lazy(() => import("./stage-contents/stage-3-content"));
const Stage4Content = lazy(() => import("./stage-contents/stage-4-content"));
const Stage5bContent = lazy(() => import("./stage-contents/stage-5b-content"));
const LLMTracePanel = lazy(() =>
  import("@/components/ui/custom/llm-trace-panel").then((module) => ({
    default: module.LLMTracePanel,
  })),
);
const SimulationViewer = lazy(() =>
  import("@/components/dag/simulation-viewer").then((module) => ({
    default: module.SimulationViewer,
  })),
);

type AnyStageData =
  | Stage0Data
  | Stage1aData
  | Stage1bData
  | Stage2Data
  | Stage3Data
  | Stage4Data
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
  const { isInvalidated, pendingStagePatches, refiningStageId, setPrefill, readOnly } =
    useRefinement();
  const invalidated = isInvalidated(stage.id);
  const effectiveStatus = resolveStageObservedStatus(status, stageRun);
  const isCompleted = effectiveStatus === "completed";
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
    effectiveStatus,
  );
  const showLogViewer = effectiveStatus !== "pending";
  const logView = showLogViewer ? (
    <StageLogView
      logs={logs}
      status={effectiveStatus}
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
      status={effectiveStatus}
      elapsedMs={elapsedMs}
      context={stage.description}
      outcome={outcome}
      loadingHint={stage.loadingHint}
      actions={
        stage.id === "stage-3" && isCompleted && projectedStageData && !readOnly ? (
          <Stage3FixAction data={projectedStageData as Stage3Data} onFix={handleFixMeasurements} />
        ) : undefined
      }
      runningContent={
        stage.id === "stage-2" && effectiveStatus === "running" ? (
          <Suspense fallback={null}>
            <Stage2RunningContent
              workspaceId={workspaceId}
              stageStatus={effectiveStatus}
              stageRun={stageRun}
            />
          </Suspense>
        ) : stage.id === "stage-4" && effectiveStatus === "running" ? (
          <Suspense fallback={null}>
            <Stage4RunningContent
              workspaceId={workspaceId}
              stageStatus={effectiveStatus}
              stageRun={stageRun}
            />
          </Suspense>
        ) : undefined
      }
      invalidated={invalidated}
      logView={logView}
      panelContent={
        projectedStageData?.llm_trace ? (
          <Suspense fallback={null}>
            <LLMTracePanel
              trace={projectedStageData.llm_trace}
              workspaceId={workspaceId}
              stageId={stage.id}
              interactive={stage.interactive}
            />
          </Suspense>
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

function createStageDataAdapter<TData>(Component: ComponentType<{ data: TData }>) {
  return function StageDataAdapter({ data }: StageContentAdapterProps) {
    return <Component data={data as TData} />;
  };
}

function Stage4ConnectedContent({ workspaceId, data }: { workspaceId: string; data: Stage4Data }) {
  const { data: stage1b } = useStageData<Stage1bData>(workspaceId, "stage-1b", true);
  return <Stage4Content data={data} indicators={stage1b?.causal_spec.measurement.indicators} />;
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
  const { refinementMessages, selectedScenarioKey, selectScenario, readOnly } = useRefinement();
  const { data: stage1a } = useStageData<Stage1aData>(workspaceId, "stage-1a", true);
  const { data: stage1b } = useStageData<Stage1bData>(workspaceId, "stage-1b", true);
  const { data: stage4 } = useStageData<Stage4Data>(workspaceId, "stage-4", true);
  const { data: stage5b } = useStageData<Stage5bData>(workspaceId, "stage-5b", true);

  const outcomeName = useMemo(
    () => stage1a?.latent_model.constructs.find((construct) => construct.is_outcome)?.name ?? null,
    [stage1a],
  );
  // Stage 6 visualizes the estimation projection — the retained latent states plus
  // the observed known-input drivers that the SSM actually fits and simulates — not
  // the full theoretical Stage 1a model. Nodes dropped in Stage 1b (marginalized
  // root confounders, non-identifiable treatments) are therefore excluded, and known
  // inputs render as exogenous (held drivers, no self-dynamics) since they leave the
  // latent state vector.
  const graph = useMemo(() => {
    const estimation = stage1b?.causal_spec.estimation;
    const stateOrder = new Set(estimation?.state_order ?? []);
    const knownInputs = new Set((estimation?.known_inputs ?? []).map((input) => input.construct));
    const constructs = (stage1a?.latent_model.constructs ?? [])
      .filter((c) => stateOrder.has(c.name) || knownInputs.has(c.name))
      .map((c) => (knownInputs.has(c.name) ? { ...c, role: "exogenous" as const } : c));
    return {
      constructs,
      edges: estimation?.edges ?? [],
      indicators: stage1b?.causal_spec.measurement.indicators,
      edgePosteriors: buildEdgePosteriors({ stage1a, stage4, stage5b }),
    };
  }, [stage1a, stage1b, stage4, stage5b]);
  // Synthesize the trajectory / drift visuals for the data's saved scenarios
  // (each carries its own clamps + summary) against the projected estimation
  // graph, so the interactive DAG is visible on the stage-6 page until the
  // backend simulate tool joins the local loop. The trajectories are synthesized
  // client-side (not real inference output), so in production this is restricted
  // to curated shared/demo workspaces (DEMO, GOLDEN, …) and never fabricates
  // scenarios for real user analyses. Dev runs it everywhere for previewing.
  const allowMockScenarios =
    process.env.NODE_ENV !== "production" || isSharedWorkspaceId(workspaceId);
  const mockScenarios = useMemo(
    () =>
      allowMockScenarios && outcomeName && graph.constructs.length > 0
        ? synthesizeMockScenarios(
            graph.constructs,
            graph.edges,
            graph.indicators ?? [],
            outcomeName,
            data.saved_scenarios,
          )
        : null,
    [allowMockScenarios, outcomeName, graph, data.saved_scenarios],
  );
  const scenarios = useMemo(
    () =>
      buildStage6Scenarios({
        trace: data.llm_trace,
        refinementMessages: [
          ...(refinementMessages["stage-6"] ?? []),
          ...(mockScenarios ? buildDevMockMessages(mockScenarios) : []),
        ],
      }),
    [data.llm_trace, refinementMessages, mockScenarios],
  );
  const onSimulate = useMemo(
    () =>
      mockScenarios
        ? makeMockSimulate(mockScenarios.baseline.result)
        : readOnly
          ? undefined
          : createSimulateDispatch(workspaceId),
    [mockScenarios, readOnly, workspaceId],
  );

  return (
    <SimulationViewer
      scenarios={scenarios}
      graph={graph}
      selectedKey={selectedScenarioKey}
      onSelect={selectScenario}
      rankingResults={data.intervention_results}
      onSimulate={onSimulate}
    />
  );
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
