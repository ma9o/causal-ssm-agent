"use client";

import { Button } from "@/components/ui/button";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { CI_LOWER, CI_UPPER } from "@/lib/constants/diagnostics";
import { formatNumber } from "@/lib/utils/format";
import { buildHistogram, quantile } from "@/lib/utils/histogram";
import type { TreatmentEffect } from "@causal-ssm/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  ComposedChart,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

const col = createColumnHelper<TreatmentEffect>();

/** Compute mean of posterior draws, or null if unavailable. */
function meanDraws(draws: number[] | null | undefined): number | null {
  if (!draws || draws.length === 0) return null;
  return draws.reduce((a, b) => a + b, 0) / draws.length;
}

function probPositive(draws: number[] | null | undefined): number | null {
  if (!draws || draws.length === 0) return null;
  return draws.filter((draw) => draw > 0).length / draws.length;
}

function peakEffect(effect: TreatmentEffect): number | null {
  return effect.temporal?.peak_effect ?? null;
}

function timeToPeak(effect: TreatmentEffect): number | null {
  return effect.temporal?.time_to_peak_days ?? null;
}

function PosteriorHistogram({ draws, mean }: { draws: number[]; mean: number | null }) {
  if (draws.length === 0) return <span className="text-xs text-muted-foreground">--</span>;

  const bins = buildHistogram(draws, Math.min(25, Math.ceil(Math.sqrt(draws.length))));

  return (
    <div className="ml-auto h-16 w-44">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={bins} margin={{ top: 2, right: 4, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="binCenter"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v: number) => formatNumber(v, 2)}
            tick={{ fontSize: 9 }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis hide />
          <RechartsTooltip
            formatter={(v: number | string | undefined) => {
              const numeric = typeof v === "number" ? v : Number(v);
              return [Number.isFinite(numeric) ? formatNumber(numeric, 0) : "--", "count"] as const;
            }}
            labelFormatter={(l: unknown) => {
              const numeric = typeof l === "number" ? l : Number(l);
              return Number.isFinite(numeric) ? `τ = ${formatNumber(numeric, 3)}` : "τ = --";
            }}
            contentStyle={{ fontSize: 10, padding: "2px 6px" }}
          />
          <ReferenceLine x={0} strokeDasharray="4 3" className="stroke-muted-foreground/60" />
          {mean !== null && <ReferenceLine x={mean} stroke="var(--foreground)" strokeWidth={1.5} />}
          <Bar dataKey="count" fill="var(--muted-foreground)" opacity={0.35} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function InlineManifestProjection({ effect }: { effect: TreatmentEffect }) {
  const entries = Object.entries(effect.manifest_effects ?? {})
    .filter((entry): entry is [string, number] => entry[1] != null)
    .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]));
  if (entries.length === 0) return null;

  return (
    <div className="flex items-start gap-4 px-6 py-2.5">
      <span className="shrink-0 pt-px text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        Indicators
      </span>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
        {entries.map(([indicator, value]) => (
          <span key={indicator} className="inline-flex items-baseline gap-1.5">
            <span className="text-muted-foreground">{indicator}</span>
            <span className="font-mono text-xs tabular-nums">
              {value >= 0 ? "+" : ""}
              {formatNumber(value)}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

export function TreatmentRankingTable({
  results,
}: {
  results: TreatmentEffect[];
}) {
  const [selectedTreatment, setSelectedTreatment] = useState<string | null>();

  const sorted = useMemo(
    () =>
      [...results].sort(
        (a, b) => Math.abs(meanDraws(b.posterior_draws) ?? 0) - Math.abs(meanDraws(a.posterior_draws) ?? 0),
      ),
    [results],
  );

  const columns = useMemo(
    () => [
      col.accessor("treatment", {
        header: "Treatment",
        cell: (info) => <span className="font-medium">{info.getValue()}</span>,
      }),
      col.display({
        id: "effect_size",
        header: () => (
          <HeaderWithTooltip
            label="τ̂"
            tooltip="Estimated total interventional treatment effect via do-operator steady-state intervention. Positive values indicate the treatment increases the outcome."
            className="font-mono"
          />
        ),
        cell: ({ row }) => {
          const v = meanDraws(row.original.posterior_draws);
          return v === null ? "—" : formatNumber(v);
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "ci_95",
        header: () => (
          <HeaderWithTooltip
            label="95% CI"
            tooltip="95% credible interval computed from the posterior draws (2.5th–97.5th percentile)."
          />
        ),
        cell: ({ row }) => {
          const draws = row.original.posterior_draws;
          if (!draws || draws.length === 0) return "—";
          const sortedDraws = [...draws].sort((a, b) => a - b);
          const lo = quantile(sortedDraws, CI_LOWER);
          const hi = quantile(sortedDraws, CI_UPPER);
          return `[${formatNumber(lo)}, ${formatNumber(hi)}]`;
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "prob_positive",
        header: () => (
          <HeaderWithTooltip
            label="P(τ > 0)"
            tooltip="Posterior probability that the total intervention effect is positive."
            className="font-mono"
          />
        ),
        cell: ({ row }) => {
          const probability = probPositive(row.original.posterior_draws);
          return probability === null ? "—" : `${Math.round(probability * 100)}%`;
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "peak",
        header: () => (
          <HeaderWithTooltip
            label="Peak"
            tooltip="Maximum absolute forward-simulated effect over the default Stage 6 horizon."
          />
        ),
        cell: ({ row }) => {
          const peak = peakEffect(row.original);
          return peak === null ? "—" : formatNumber(peak);
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "time_to_peak",
        header: () => (
          <HeaderWithTooltip
            label="t → peak"
            tooltip="Days from intervention onset to the peak absolute effect."
            className="font-mono"
          />
        ),
        cell: ({ row }) => {
          const days = timeToPeak(row.original);
          return days === null ? "—" : `${formatNumber(days, 1)}d`;
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "posterior",
        header: () => (
          <HeaderWithTooltip
            label="Posterior"
            tooltip="Full posterior distribution of the treatment effect. The vertical dashed line marks zero — draws to the right indicate a positive effect."
          />
        ),
        cell: ({ row }) => {
          const draws = row.original.posterior_draws;
          return draws && draws.length > 0 ? (
            <PosteriorHistogram draws={draws} mean={meanDraws(draws)} />
          ) : (
            "—"
          );
        },
        meta: {
          align: "right",
        },
      }),
      col.display({
        id: "projection",
        header: () => (
          <HeaderWithTooltip
            label="Indicators"
            tooltip="Expandable indicator-level projection of the outcome effect through the measurement loadings."
          />
        ),
        cell: ({ row }) => {
          const manifestCount = Object.keys(row.original.manifest_effects ?? {}).length;
          if (manifestCount === 0) {
            return <span className="text-muted-foreground">—</span>;
          }
          const isSelected = row.original.treatment === selectedTreatment;
          return (
            <Button
              variant="ghost"
              size="xs"
              onClick={() =>
                setSelectedTreatment(isSelected ? null : row.original.treatment)
              }
            >
              {isSelected ? (
                <ChevronUp className="size-3" />
              ) : (
                <ChevronDown className="size-3" />
              )}
              {isSelected ? "Hide" : "Show"} {manifestCount}
            </Button>
          );
        },
        meta: {
          align: "right",
        },
      }),
    ],
    [selectedTreatment],
  );

  return (
    <InfoTable
      columns={columns as ColumnDef<TreatmentEffect, unknown>[]}
      data={sorted}
      compact
      estimateRowHeight={68}
      rowClassName={(row, index) => {
        if (row.treatment === selectedTreatment) {
          return "bg-muted/40";
        }
        return index === 0 ? "bg-emerald-500/10" : undefined;
      }}
      isRowExpanded={(row) =>
        row.treatment === selectedTreatment &&
        Object.keys(row.manifest_effects ?? {}).length > 0
      }
      renderExpandedRow={(row) => <InlineManifestProjection effect={row} />}
    />
  );
}
