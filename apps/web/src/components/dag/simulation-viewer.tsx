"use client";

import type { CausalEdge, Construct, Indicator, TreatmentEffect } from "@nof1-causal-lab/api-types";
import { Bot, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Stage6Scenario } from "@/components/pipeline/stage-contents/stage-6-scenarios";
import {
  ManifestProjection,
  PosteriorHistogram,
} from "@/components/stages/inference/treatment-effect-visuals";
import { TreatmentRankingTable } from "@/components/stages/inference/treatment-ranking-table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { formatNumber } from "@/lib/utils/format";
import { EffectSummary } from "./effect-summary";
import { InterventionDag } from "./intervention-dag";
import { formatClampReferenceLabel, formatClampShortLabel } from "./intervention-dag-semantics";
import type { EdgePosterior, Stage6SimulationResult } from "./intervention-dag-types";
import type { DagMetric, StaticScenarioInput } from "./intervention-dag-view-model";
import { ScenarioHeader } from "./query-header";
import { ScenarioRail } from "./scenario-rail";
import { TrajectoryChart } from "./trajectory-chart";
import { ViewControls } from "./view-controls";

export interface SimulationViewerGraph {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  edgePosteriors?: Record<string, EdgePosterior>;
}

export interface SimulationViewerProps {
  scenarios: Stage6Scenario[];
  graph: SimulationViewerGraph;
  finalSummary?: string | null;
  selectedKey?: string | null;
  onSelect?: (key: string) => void;
  dagHeight?: string;
  /** Raw baseline ranking, surfaced as a collapsed dense comparison table. */
  rankingResults?: TreatmentEffect[];
}

function hasTrajectories(result: Stage6SimulationResult): boolean {
  return Boolean(
    result.effect_trajectory &&
      result.effect_trajectory.length > 0 &&
      result.visualization?.node_effect_trajectories,
  );
}

function staticScenarioFor(scenario: Stage6Scenario): StaticScenarioInput {
  if (scenario.provenance === "baseline") {
    return {
      treatment: scenario.treatment,
      outcome: scenario.outcome,
      effectMagnitude: scenario.summary.mean,
      actionLabelShort: "shift +1.0",
      actionReferenceLabel: "from baseline",
    };
  }
  const clamp = scenario.result.clamps[0];
  return {
    treatment: clamp?.variable ?? scenario.outcome,
    outcome: scenario.outcome,
    effectMagnitude: scenario.summary.mean,
    actionLabelShort: clamp ? formatClampShortLabel(clamp) : "clamp",
    actionReferenceLabel: clamp ? formatClampReferenceLabel(scenario.result, clamp) : "",
  };
}

function NarrativeBlock({ content }: { content: string }) {
  return (
    <div className="rounded-lg border bg-muted/20 p-4">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        <Bot className="h-3.5 w-3.5" />
        Stage Interpretation
      </div>
      <div
        className="prose prose-sm max-w-none overflow-y-auto text-sm [&_p]:my-2 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0"
        style={{ maxHeight: "12.5rem" }}
      >
        <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
      </div>
    </div>
  );
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${formatNumber(value)}`;
}

function TemporalSnapshots({
  temporal,
}: {
  temporal: NonNullable<Extract<Stage6Scenario, { provenance: "baseline" }>["temporal"]>;
}) {
  const points: { label: string; value: number }[] = [
    { label: "1d", value: temporal.effect_1d },
    { label: "7d", value: temporal.effect_7d },
    { label: "30d", value: temporal.effect_30d },
    { label: "peak", value: temporal.peak_effect },
  ];
  return (
    <div className="rounded-lg border bg-muted/20 p-3">
      <div className="mb-2 text-[11px] uppercase tracking-wide text-muted-foreground">
        Forward effect
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-1">
        {points.map((point) => (
          <div key={point.label} className="space-y-0.5">
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
              {point.label}
            </div>
            <div className="font-mono text-sm tabular-nums">{signed(point.value)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SimulationWarnings({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <div className="flex items-start gap-2 rounded-lg border border-amber-400/40 bg-amber-400/5 p-3 text-xs text-amber-700 dark:text-amber-400">
      <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <ul className="space-y-0.5">
        {warnings.map((warning) => (
          <li key={warning}>{warning}</li>
        ))}
      </ul>
    </div>
  );
}

function ScenarioDetail({
  scenario,
  graph,
  metric,
  onMetricChange,
  dagHeight,
}: {
  scenario: Stage6Scenario;
  graph: SimulationViewerGraph;
  metric: DagMetric;
  onMetricChange: (metric: DagMetric) => void;
  dagHeight: string;
}) {
  const animated = scenario.provenance === "simulation" && hasTrajectories(scenario.result);

  return (
    <div className="space-y-3">
      <ScenarioHeader scenario={scenario} />
      {animated ? <ViewControls metric={metric} onMetricChange={onMetricChange} /> : null}
      {animated ? (
        <InterventionDag
          constructs={graph.constructs}
          edges={graph.edges}
          indicators={graph.indicators}
          edgePosteriors={graph.edgePosteriors}
          simulationResult={scenario.result}
          metric={metric}
          requestedHorizonDays={scenario.requestedHorizonDays}
          height={dagHeight}
        />
      ) : (
        <InterventionDag
          constructs={graph.constructs}
          edges={graph.edges}
          indicators={graph.indicators}
          edgePosteriors={graph.edgePosteriors}
          staticScenario={staticScenarioFor(scenario)}
          height={dagHeight}
        />
      )}
      <EffectSummary scenario={scenario} />
      {animated ? <TrajectoryChart result={scenario.result} metric={metric} /> : null}
      {scenario.provenance === "simulation" ? (
        <SimulationWarnings warnings={scenario.result.warnings} />
      ) : null}
      {scenario.provenance === "baseline" ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {scenario.posteriorDraws && scenario.posteriorDraws.length > 0 ? (
            <div className="rounded-lg border bg-muted/20 p-3">
              <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">
                Posterior distribution
              </div>
              <PosteriorHistogram
                draws={scenario.posteriorDraws}
                mean={scenario.summary.mean}
                className="h-40 w-full"
              />
            </div>
          ) : null}
          {scenario.temporal ? <TemporalSnapshots temporal={scenario.temporal} /> : null}
        </div>
      ) : null}
      {scenario.manifestEffects ? (
        <ManifestProjection
          manifestEffects={scenario.manifestEffects}
          className="rounded-lg border bg-muted/20 p-3"
        />
      ) : null}
    </div>
  );
}

/**
 * Stage 6 simulation viewer. The left (content) column of the stage: an opening
 * narrative, a rail of materialized scenarios (baseline ranking + chat-minted
 * simulations), and an adaptive detail view for the focused scenario. The chat
 * that mints new scenarios lives in the shell's trace pane and shares selection
 * via RefinementContext.
 */
export function SimulationViewer({
  scenarios,
  graph,
  finalSummary,
  selectedKey,
  onSelect,
  dagHeight = "560px",
  rankingResults,
}: SimulationViewerProps) {
  const [metric, setMetric] = useState<DagMetric>("effect");

  const selected = useMemo(
    () => scenarios.find((scenario) => scenario.key === selectedKey) ?? scenarios[0] ?? null,
    [scenarios, selectedKey],
  );

  return (
    <div className="space-y-4">
      {finalSummary?.trim() ? <NarrativeBlock content={finalSummary} /> : null}
      {scenarios.length === 0 ? (
        <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
          No treatment effects were estimated. This may happen if no treatments passed
          identification checks.
        </div>
      ) : (
        <>
          <ScenarioRail
            scenarios={scenarios}
            selectedKey={selected?.key ?? null}
            onSelect={(key) => onSelect?.(key)}
          />
          {selected ? (
            <ScenarioDetail
              scenario={selected}
              graph={graph}
              metric={metric}
              onMetricChange={setMetric}
              dagHeight={dagHeight}
            />
          ) : null}
        </>
      )}
      {rankingResults && rankingResults.length > 0 ? (
        <Accordion>
          <AccordionItem value="all-treatments">
            <AccordionTrigger className="text-sm">
              All treatments (baseline ranking)
            </AccordionTrigger>
            <AccordionContent>
              <TreatmentRankingTable results={rankingResults} />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      ) : null}
    </div>
  );
}
