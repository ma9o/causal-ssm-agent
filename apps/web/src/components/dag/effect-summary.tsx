"use client";

import type { Stage6Scenario } from "@/components/pipeline/stage-contents/stage-6-scenarios";
import { formatNumber } from "@/lib/utils/format";

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${formatNumber(value)}`;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="space-y-0.5">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="font-mono text-sm">{value}</div>
      {hint ? <div className="text-[10px] text-muted-foreground">{hint}</div> : null}
    </div>
  );
}

/** Posterior summary for the focused scenario, common to both provenances. */
export function EffectSummary({ scenario }: { scenario: Stage6Scenario }) {
  const { summary, outcome } = scenario;

  const fourth =
    scenario.provenance === "simulation"
      ? { label: "Reference mean", value: formatNumber(scenario.result.reference_mean) }
      : { label: "Posterior draws", value: `${scenario.posteriorDraws?.length ?? 0}` };

  return (
    <div className="grid grid-cols-2 gap-4 rounded-lg border bg-muted/20 p-3 sm:grid-cols-4">
      <Stat
        label={`Effect on ${outcome}`}
        value={`${signed(summary.mean)} SD`}
        hint={`95% CI [${formatNumber(summary.lower95)}, ${formatNumber(summary.upper95)}]`}
      />
      <Stat label="P(effect > 0)" value={`${Math.round(summary.probPositive * 100)}%`} />
      <Stat
        label="Peak effect"
        value={summary.peakEffect == null ? "—" : `${signed(summary.peakEffect)} SD`}
        hint={
          summary.timeToPeakDays == null
            ? undefined
            : `day ${formatNumber(summary.timeToPeakDays, 1)}`
        }
      />
      <Stat label={fourth.label} value={fourth.value} />
    </div>
  );
}
