"use client";

import type { Stage6Scenario } from "@/components/pipeline/stage-contents/stage-6-scenarios";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { formatNumber } from "@/lib/utils/format";

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${formatNumber(value)}`;
}

function provenanceLabel(scenario: Stage6Scenario): string {
  if (scenario.provenance === "baseline") return "Baseline";
  return scenario.result.start.kind === "abducted" ? "Counterfactual" : "Interventional";
}

function ScenarioCard({
  scenario,
  selected,
  onSelect,
}: {
  scenario: Stage6Scenario;
  selected: boolean;
  onSelect: () => void;
}) {
  const positive = scenario.summary.mean >= 0;
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        "flex w-44 shrink-0 snap-start flex-col gap-1.5 rounded-lg border p-2.5 text-left transition-all",
        "hover:border-foreground/40 hover:shadow-sm",
        selected ? "border-primary bg-primary/5 ring-2 ring-primary/30" : "border-border bg-card",
      )}
    >
      <div className="flex items-center justify-between gap-1">
        <Badge
          variant={scenario.provenance === "baseline" ? "outline" : "secondary"}
          className="px-1.5 py-0 text-[10px]"
        >
          {provenanceLabel(scenario)}
        </Badge>
        <span className="text-[10px] text-muted-foreground">
          P&gt;0 {Math.round(scenario.summary.probPositive * 100)}%
        </span>
      </div>
      <div className="truncate font-mono text-[11px] text-foreground" title={scenario.title}>
        {scenario.title}
      </div>
      <div className="flex items-baseline justify-between gap-1">
        <span
          className={cn(
            "font-mono text-base font-semibold tabular-nums",
            positive ? "text-teal-600 dark:text-teal-400" : "text-rose-600 dark:text-rose-400",
          )}
        >
          {signed(scenario.summary.mean)}
        </span>
        <span className="truncate text-[10px] text-muted-foreground" title={scenario.outcome}>
          → {scenario.outcome}
        </span>
      </div>
    </button>
  );
}

/**
 * Horizontal rail of selectable scenario cards (one in focus). Materialized
 * simulations come first (newest first), followed by the baseline ranking.
 */
export function ScenarioRail({
  scenarios,
  selectedKey,
  onSelect,
}: {
  scenarios: Stage6Scenario[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  if (scenarios.length === 0) {
    return null;
  }
  return (
    <div className="flex snap-x gap-2 overflow-x-auto pb-1">
      {scenarios.map((scenario) => (
        <ScenarioCard
          key={scenario.key}
          scenario={scenario}
          selected={scenario.key === selectedKey}
          onSelect={() => onSelect(scenario.key)}
        />
      ))}
    </div>
  );
}
