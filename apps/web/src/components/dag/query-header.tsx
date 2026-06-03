"use client";

import type { ReactNode } from "react";
import type {
  BaselineScenario,
  SimulationScenario,
  Stage6Scenario,
} from "@/components/pipeline/stage-contents/stage-6-scenarios";
import { Badge } from "@/components/ui/badge";

function MetaChip({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-md border bg-background px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
      {children}
    </span>
  );
}

function SimulationHeader({ scenario }: { scenario: SimulationScenario }) {
  const { result } = scenario;
  const abducted = result.start.kind === "abducted";
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant={abducted ? "outline" : "secondary"}>
          {abducted ? "Counterfactual · abducted start" : "Interventional · baseline start"}
        </Badge>
        <MetaChip>{scenario.title}</MetaChip>
        <MetaChip>→ {scenario.outcome}</MetaChip>
        <MetaChip>{result.estimand}</MetaChip>
        {scenario.requestedHorizonDays ? (
          <MetaChip>{scenario.requestedHorizonDays}-day</MetaChip>
        ) : null}
        {abducted && result.start.time ? (
          <MetaChip>from {result.start.time.slice(0, 10)}</MetaChip>
        ) : null}
      </div>
      <p className="text-[11px] text-muted-foreground">
        These define the simulation — change them by asking in chat; each answer mints a new
        scenario.
      </p>
    </div>
  );
}

function BaselineHeader({ scenario }: { scenario: BaselineScenario }) {
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant="outline">Baseline · rung 2</Badge>
        <MetaChip>do({scenario.treatment} += 1 SD)</MetaChip>
        <MetaChip>→ {scenario.outcome}</MetaChip>
        <MetaChip>steady state</MetaChip>
      </div>
      <p className="text-[11px] text-muted-foreground">
        Pre-computed from the baseline ranking. Ask in chat to explore a custom intervention amount,
        horizon, or counterfactual.
      </p>
    </div>
  );
}

/** Read-only chips describing what defines the focused scenario. */
export function ScenarioHeader({ scenario }: { scenario: Stage6Scenario }) {
  return scenario.provenance === "simulation" ? (
    <SimulationHeader scenario={scenario} />
  ) : (
    <BaselineHeader scenario={scenario} />
  );
}
