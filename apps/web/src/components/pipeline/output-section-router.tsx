"use client";

import type {
  ArtifactViewId,
  ArtifactViewData as ArtifactViewPayload,
  RawDataData,
  LatentStructureData,
  MeasurementStructureViewData,
  MeasurementsData,
  ValidationReportData,
  StatisticalModelSpecData,
  PosteriorData,
  BaselineReportData,
  TransitionMeta,
} from "@nof1-causal-lab/api-types";
import { type ComponentType, lazy, memo, type ReactNode, Suspense, useMemo } from "react";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import type { AnalysisTransitionRun } from "@/lib/api/analysis";
import { createSimulateDispatch } from "@/components/dag/interactive/dispatch-simulate";
import {
  buildDevMockMessages,
  makeMockSimulate,
  synthesizeMockScenarios,
} from "@/components/dag/interactive/dev-mock-scenario";
import { useWorkspaceView } from "@/lib/contexts/workspace-view-context";
import type { TransitionRunStatus, TransitionTiming } from "@/lib/hooks/use-run-events";
import { useArtifactView } from "@/lib/hooks/use-artifact-view";
import { useLLMTrace } from "@/lib/hooks/use-llm-trace";
import { resolveTransitionObservedStatus } from "@/lib/transition-runtime";
import {
  buildEdgePosteriors,
  buildBaselineReportScenarios,
} from "./output-views/baseline-report-scenarios";
import { OutputPresentationShell } from "./output-presentation-shell";

const RawDataView = lazy(() => import("./output-views/raw-data-view"));
const LatentStructureView = lazy(() => import("./output-views/latent-structure-view"));
const MeasurementStructureView = lazy(() => import("./output-views/measurement-structure-view"));
const MeasurementsView = lazy(() => import("./output-views/measurements-view"));
const MeasurementsRunningOutputView = lazy(
  () => import("./output-views/measurements-running-view"),
);
const StatisticalModelSpecRunningOutputView = lazy(
  () => import("./output-views/statistical-model-spec-running-view"),
);
const ValidationReportView = lazy(() => import("./output-views/validation-report-view"));
const StatisticalModelSpecView = lazy(() => import("./output-views/statistical-model-spec-view"));
const PosteriorView = lazy(() => import("./output-views/posterior-view"));
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

type ArtifactViewData = ArtifactViewPayload & {
  context?: string;
  llm_trace_ref?: string | null;
};

type OutputSectionRouterProps = {
  output: TransitionMeta;
  workspaceId: string;
  status: TransitionRunStatus;
  timing?: TransitionTiming;
  transitionRun?: AnalysisTransitionRun;
  errorMessage?: string;
  staleArtifactIds?: string[];
};

function transitionRunsEqual(
  previous?: AnalysisTransitionRun,
  next?: AnalysisTransitionRun,
): boolean {
  return (
    previous?.execution?.stateType === next?.execution?.stateType &&
    previous?.execution?.startTime === next?.execution?.startTime &&
    previous?.execution?.endTime === next?.execution?.endTime
  );
}

function OutputSectionRouterInner({
  output,
  workspaceId,
  status,
  timing,
  transitionRun,
  errorMessage,
  staleArtifactIds,
}: OutputSectionRouterProps) {
  const effectiveStatus = resolveTransitionObservedStatus(status, transitionRun);
  const isCompleted = effectiveStatus === "completed";
  const elapsedMs =
    timing?.completedAt && timing?.startedAt ? timing.completedAt - timing.startedAt : undefined;

  // Read context + trace from the artifact data once the output has completed.
  const { data: artifactData } = useArtifactView<ArtifactViewData>(
    workspaceId,
    output.id,
    isCompleted,
  );
  const traceRef = artifactData?.llm_trace_ref ?? null;
  const { data: llmTrace } = useLLMTrace(workspaceId, traceRef, isCompleted && !!traceRef);

  return (
    <OutputPresentationShell
      output={output}
      status={effectiveStatus}
      elapsedMs={elapsedMs}
      context={output.description}
      errorMessage={errorMessage}
      staleArtifactIds={staleArtifactIds}
      loadingHint={output.loadingHint}
      runningContent={
        output.id === "measurements" && effectiveStatus === "running" ? (
          <Suspense fallback={null}>
            <MeasurementsRunningOutputView workspaceId={workspaceId} />
          </Suspense>
        ) : output.id === "statistical_model_spec" &&
          (effectiveStatus === "running" || effectiveStatus === "failed") ? (
          <Suspense fallback={null}>
            <StatisticalModelSpecRunningOutputView
              workspaceId={workspaceId}
              showError={effectiveStatus !== "failed"}
            />
          </Suspense>
        ) : undefined
      }
      panelContent={
        llmTrace ? (
          <Suspense fallback={null}>
            <LLMTracePanel trace={llmTrace} />
          </Suspense>
        ) : undefined
      }
    >
      {isCompleted && (
        <ErrorBoundary>
          <Suspense fallback={null}>
            <OutputView artifactId={output.id} workspaceId={workspaceId} data={artifactData} />
          </Suspense>
        </ErrorBoundary>
      )}
    </OutputPresentationShell>
  );
}

export const OutputSectionRouter = memo(
  OutputSectionRouterInner,
  (previous, next) =>
    previous.workspaceId === next.workspaceId &&
    previous.output.id === next.output.id &&
    previous.status === next.status &&
    previous.timing?.startedAt === next.timing?.startedAt &&
    previous.timing?.completedAt === next.timing?.completedAt &&
    previous.errorMessage === next.errorMessage &&
    (previous.staleArtifactIds?.join("|") ?? "") === (next.staleArtifactIds?.join("|") ?? "") &&
    transitionRunsEqual(previous.transitionRun, next.transitionRun),
);

type OutputViewAdapterProps = {
  workspaceId: string;
  data: ArtifactViewData;
};

function createArtifactDataAdapter<TData>(Component: ComponentType<{ data: TData }>) {
  return function ArtifactDataAdapter({ data }: OutputViewAdapterProps) {
    return <Component data={data as TData} />;
  };
}

function ModelSpecConnectedContent({
  workspaceId,
  data,
}: {
  workspaceId: string;
  data: StatisticalModelSpecData;
}) {
  const { data: measurementStructure } = useArtifactView<MeasurementStructureViewData>(
    workspaceId,
    "measurement_structure",
    true,
  );
  return (
    <StatisticalModelSpecView
      data={data}
      indicators={measurementStructure?.causal_design.measurement.indicators}
    />
  );
}

function PosteriorConnectedContent({
  workspaceId,
  data,
}: {
  workspaceId: string;
  data: PosteriorData;
}) {
  return <PosteriorView workspaceId={workspaceId} data={data} />;
}

function BaselineReportConnectedContent({
  workspaceId,
  data,
}: {
  workspaceId: string;
  data: BaselineReportData;
}) {
  const { selectedScenarioKey, selectScenario, readOnly } = useWorkspaceView();
  const { data: latentStructure } = useArtifactView<LatentStructureData>(
    workspaceId,
    "latent_structure",
    true,
  );
  const { data: measurementStructure } = useArtifactView<MeasurementStructureViewData>(
    workspaceId,
    "measurement_structure",
    true,
  );
  const { data: modelSpec } = useArtifactView<StatisticalModelSpecData>(
    workspaceId,
    "statistical_model_spec",
    true,
  );
  const { data: posterior } = useArtifactView<PosteriorData>(workspaceId, "posterior", true);
  const { data: llmTrace } = useLLMTrace(
    workspaceId,
    data.llm_trace_ref ?? null,
    !!data.llm_trace_ref,
  );

  const outcomeName = useMemo(
    () =>
      latentStructure?.latent_structure.constructs.find((construct) => construct.is_outcome)
        ?.name ?? null,
    [latentStructure],
  );
  // analysis visualizes the estimation projection — the retained latent states plus
  // the observed known-input drivers that the SSM actually fits and simulates — not
  // the full theoretical latent-structure model. Nodes dropped in measurement-structure (marginalized
  // root confounders, non-identifiable treatments) are therefore excluded, and known
  // inputs render as exogenous (held drivers, no self-dynamics) since they leave the
  // latent state vector.
  const graph = useMemo(() => {
    const estimation = measurementStructure?.causal_design.estimation;
    const stateOrder = new Set(estimation?.state_order ?? []);
    const knownInputs = new Set((estimation?.known_inputs ?? []).map((input) => input.construct));
    const constructs = (latentStructure?.latent_structure.constructs ?? [])
      .filter((c) => stateOrder.has(c.name) || knownInputs.has(c.name))
      .map((c) => (knownInputs.has(c.name) ? { ...c, role: "exogenous" as const } : c));
    return {
      constructs,
      edges: estimation?.edges ?? [],
      indicators: measurementStructure?.causal_design.measurement.indicators,
      edgePosteriors: buildEdgePosteriors({ latentStructure, modelSpec, posterior }),
    };
  }, [latentStructure, measurementStructure, modelSpec, posterior]);
  // Synthesize the trajectory / drift visuals for the data's saved scenarios
  // (each carries its own clamps + summary) against the projected estimation
  // graph, so the interactive DAG is visible on the baseline_report page until the
  // backend simulate tool joins the local loop. The trajectories are synthesized
  // client-side (not real inference output), so in production this is restricted
  // to read-only (published/demo) workspaces, where live simulate is unavailable
  // anyway — it never fabricates scenarios for a live user analysis. Dev runs it
  // everywhere for previewing.
  const allowMockScenarios = process.env.NODE_ENV !== "production" || readOnly;
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
      buildBaselineReportScenarios({
        trace: llmTrace,
        extraMessages: mockScenarios ? buildDevMockMessages(mockScenarios) : [],
      }),
    [llmTrace, mockScenarios],
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

const outputViewAdapters = {
  raw_data: ({ workspaceId, data }: OutputViewAdapterProps) => (
    <RawDataView workspaceId={workspaceId} data={data as RawDataData} />
  ),
  latent_structure: createArtifactDataAdapter<LatentStructureData>(LatentStructureView),
  measurement_structure:
    createArtifactDataAdapter<MeasurementStructureViewData>(MeasurementStructureView),
  measurements: ({ workspaceId, data }: OutputViewAdapterProps) => (
    <MeasurementsView workspaceId={workspaceId} data={data as MeasurementsData} />
  ),
  validation_report: createArtifactDataAdapter<ValidationReportData>(ValidationReportView),
  statistical_model_spec: ({ workspaceId, data }: OutputViewAdapterProps) => (
    <ModelSpecConnectedContent workspaceId={workspaceId} data={data as StatisticalModelSpecData} />
  ),
  posterior: ({ workspaceId, data }: OutputViewAdapterProps) => (
    <PosteriorConnectedContent workspaceId={workspaceId} data={data as PosteriorData} />
  ),
  baseline_report: ({ workspaceId, data }: OutputViewAdapterProps) => (
    <BaselineReportConnectedContent workspaceId={workspaceId} data={data as BaselineReportData} />
  ),
} satisfies Record<ArtifactViewId, (props: OutputViewAdapterProps) => ReactNode>;

function OutputView({
  artifactId,
  workspaceId,
  data,
}: {
  artifactId: ArtifactViewId;
  workspaceId: string;
  data?: ArtifactViewData;
}) {
  if (!data) return null;
  const renderOutputView = outputViewAdapters[artifactId];
  return renderOutputView ? renderOutputView({ workspaceId, data }) : null;
}
