"use client";

import type { CausalEdge, Construct, Indicator, TreatmentEffect } from "@nof1-causal-lab/api-types";
import { Bot, TriangleAlert } from "lucide-react";
import { useMemo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Stage6Scenario } from "@/components/pipeline/stage-contents/stage-6-scenarios";
import { ManifestProjection } from "@/components/stages/inference/treatment-effect-visuals";
import { TreatmentRankingTable } from "@/components/stages/inference/treatment-ranking-table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { InteractiveDag } from "./interactive/interactive-dag";
import type { SimulateFn } from "./interactive/simulate-input";
import type { EdgePosterior } from "./intervention-dag-types";
import { ScenarioRail } from "./scenario-rail";

export interface SimulationViewerGraph {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  edgePosteriors?: Record<string, EdgePosterior>;
}

export interface SimulationViewerProps {
  scenarios: Stage6Scenario[];
  graph: SimulationViewerGraph;
  selectedKey?: string | null;
  onSelect?: (key: string) => void;
  /** Raw baseline ranking, surfaced as a collapsed dense comparison table. */
  rankingResults?: TreatmentEffect[];
  /** Live simulate seam; when present, do() editing is enabled on the DAG. */
  onSimulate?: SimulateFn;
}

/**
 * The LLM's explanation produced with the focused scenario — reasoning behind the
 * intervention and what the simulation shows. Sits directly under the carousel.
 */
function ScenarioBlurb({ scenario }: { scenario: Stage6Scenario }) {
  if (!scenario.blurb?.trim()) {
    return null;
  }
  return (
    <div className="flex gap-3 rounded-lg border bg-muted/20 p-4">
      <Bot className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
      <div className="prose prose-sm max-w-none text-sm [&_p]:my-1.5 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0">
        <Markdown remarkPlugins={[remarkGfm]}>{scenario.blurb}</Markdown>
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
  onSimulate,
}: {
  scenario: Stage6Scenario;
  graph: SimulationViewerGraph;
  onSimulate?: SimulateFn;
}) {
  return (
    <div className="space-y-3">
      <InteractiveDag
        constructs={graph.constructs}
        edges={graph.edges}
        indicators={graph.indicators}
        result={scenario.result}
        onSimulate={onSimulate}
      />
      <SimulationWarnings warnings={scenario.result.warnings} />
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
 * Stage 6 simulation viewer. The left (content) column of the stage: a rail of
 * scenarios (the no-intervention baseline first, then chat-minted interventions),
 * the LLM's blurb for the focused scenario, and the living DAG. The chat that
 * mints new scenarios lives in the shell's trace pane and shares selection via
 * RefinementContext.
 */
export function SimulationViewer({
  scenarios,
  graph,
  selectedKey,
  onSelect,
  rankingResults,
  onSimulate,
}: SimulationViewerProps) {
  const selected = useMemo(
    () => scenarios.find((scenario) => scenario.key === selectedKey) ?? scenarios[0] ?? null,
    [scenarios, selectedKey],
  );

  return (
    <div className="space-y-4">
      {scenarios.length === 0 ? (
        <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
          No scenarios available yet. Ask in chat to simulate an intervention.
        </div>
      ) : (
        <>
          <ScenarioRail
            scenarios={scenarios}
            selectedKey={selected?.key ?? null}
            onSelect={(key) => onSelect?.(key)}
          />
          {selected ? <ScenarioBlurb scenario={selected} /> : null}
          {selected ? (
            <ScenarioDetail scenario={selected} graph={graph} onSimulate={onSimulate} />
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
