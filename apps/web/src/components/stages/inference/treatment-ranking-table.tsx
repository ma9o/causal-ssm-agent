"use client";

import { useMemo } from "react";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import { formatNumber, formatPercent } from "@/lib/utils/format";
import { buildHistogram } from "@/lib/utils/histogram";
import type { TreatmentEffect } from "@causal-ssm/api-types";
import {
  Bar,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

const col = createColumnHelper<TreatmentEffect>();

/** Fail if not identifiable, warn if prior-sensitive. */
function effectSeverity(row: TreatmentEffect): "fail" | "warn" | undefined {
  if (!row.identifiable) return "fail";
  if (row.prior_sensitivity_warning) return "warn";
  return undefined;
}

function PosteriorHistogram({ draws, mean }: { draws: number[]; mean: number | null }) {
  if (draws.length === 0) return <span className="text-xs text-muted-foreground">--</span>;

  const bins = buildHistogram(draws, Math.min(25, Math.ceil(Math.sqrt(draws.length))));

  return (
    <div className="mx-auto h-16 w-44">
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
          {mean !== null && (
            <ReferenceLine x={mean} stroke="var(--foreground)" strokeWidth={1.5} />
          )}
          <Bar dataKey="count" fill="var(--muted-foreground)" opacity={0.35} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

const columns = [
  col.accessor("treatment", {
    header: "Treatment",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("effect_size", {
    header: () => (
      <HeaderWithTooltip
        label={"\u03C4\u0302"}
        tooltip="Estimated individual treatment effect (ITE) via do-operator steady-state intervention. Positive values indicate the treatment increases the outcome."
      />
    ),
    cell: (info) => {
      const v = info.getValue();
      return v === null ? "—" : formatNumber(v);
    },
    meta: {
      align: "right",
      mono: true,
      severity: (_v: number | null, row: TreatmentEffect) => effectSeverity(row),
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
        <PosteriorHistogram draws={draws} mean={row.original.effect_size} />
      ) : "—";
    },
  }),
  col.accessor("prob_positive", {
    header: () => (
      <HeaderWithTooltip
        label={`P(\u03C4>0)`}
        tooltip="Posterior probability that the treatment effect is positive. Values near 1 or 0 indicate strong directional evidence."
      />
    ),
    cell: (info) => {
      const p = info.getValue();
      return p == null ? "—" : formatPercent(p);
    },
    meta: {
      align: "right",
      mono: true,
      severity: (_v: number | null, row: TreatmentEffect) => effectSeverity(row),
    },
  }),
];

export function TreatmentRankingTable({ results }: { results: TreatmentEffect[] }) {
  const sorted = useMemo(
    () =>
      [...results].sort(
        (a, b) => Math.abs(b.effect_size ?? 0) - Math.abs(a.effect_size ?? 0),
      ),
    [results],
  );

  return (
    <InfoTable
      columns={columns as ColumnDef<TreatmentEffect, unknown>[]}
      data={sorted}
      estimateRowHeight={88}
      rowClassName={(_row, index) => (index === 0 ? "bg-emerald-500/10" : undefined)}
    />
  );
}
