"use client";

import { Badge } from "@/components/ui/badge";
import { InfoTable } from "@/components/ui/info-table";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { formatNumber } from "@/lib/utils/format";
import { buildHistogram } from "@/lib/utils/histogram";
import type { Extraction, LikelihoodSpec } from "@causal-ssm/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

// ── Row type ──────────────────────────────────────────────

export interface MeasurementRow {
  likelihood: LikelihoodSpec;
  extractions: Extraction[];
  priorSamples?: number[];
}

function buildCountFrequency(values: number[]): Array<{ binCenter: number; count: number }> {
  const freq = new Map<number, number>();
  for (const v of values) {
    freq.set(v, (freq.get(v) ?? 0) + 1);
  }
  return Array.from(freq.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([k, count]) => ({ binCenter: k, count }));
}

function binPriorSamples(
  priorSamples: number[],
  dataBins: Array<{ binCenter: number }>,
  nData: number,
  isDiscrete: boolean,
): Array<{ binCenter: number; prior: number }> {
  if (priorSamples.length === 0 || dataBins.length === 0) return [];

  if (isDiscrete) {
    const freq = new Map<number, number>();
    for (const v of priorSamples) {
      freq.set(v, (freq.get(v) ?? 0) + 1);
    }
    const scale = nData / priorSamples.length;
    return dataBins.map((b) => ({
      binCenter: b.binCenter,
      prior: (freq.get(b.binCenter) ?? 0) * scale,
    }));
  }

  if (dataBins.length < 2) return [];
  const binWidth = dataBins[1].binCenter - dataBins[0].binCenter;
  const firstEdge = dataBins[0].binCenter - binWidth / 2;

  const counts = new Array(dataBins.length).fill(0);
  for (const v of priorSamples) {
    const idx = Math.min(Math.max(Math.floor((v - firstEdge) / binWidth), 0), dataBins.length - 1);
    counts[idx]++;
  }

  const scale = nData / priorSamples.length;
  return dataBins.map((b, i) => ({
    binCenter: b.binCenter,
    prior: counts[i] * scale,
  }));
}

// ── Link label helper ─────────────────────────────────────

function linkLabel(link: string): string {
  switch (link) {
    case "identity":
      return "E[y] = \u03BC";
    case "log":
      return "E[y] = exp(\u03BC)";
    case "logit":
      return "E[y] = \u03C3(\u03BC)";
    case "probit":
      return "E[y] = \u03A6(\u03BC)";
    default:
      return "g\u207B\u00B9(\u03BC)";
  }
}

// ── Inline chart ──────────────────────────────────────────

function MeasurementSparkline({ row }: { row: MeasurementRow }) {
  const numericValues = row.extractions
    .map((e) => (typeof e.value === "boolean" ? (e.value ? 1 : 0) : Number(e.value)))
    .filter((v) => !Number.isNaN(v));

  if (numericValues.length === 0) return <span className="text-xs text-muted-foreground">--</span>;

  const isDiscrete =
    row.likelihood.distribution === "poisson" || row.likelihood.distribution === "bernoulli";

  const bins = isDiscrete
    ? buildCountFrequency(numericValues)
    : buildHistogram(numericValues, Math.min(15, Math.ceil(Math.sqrt(numericValues.length))));

  const prior =
    row.priorSamples && row.priorSamples.length > 0
      ? binPriorSamples(row.priorSamples, bins, numericValues.length, isDiscrete)
      : [];

  const hasPrior = prior.length > 0;

  const chartData = bins.map((b) => {
    const p = prior.find((o) => o.binCenter === b.binCenter);
    return { ...b, ...(hasPrior ? { prior: p?.prior ?? 0 } : {}) };
  });

  return (
    <div className="h-20 w-48">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 2, right: 4, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="binCenter"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v: number) => formatNumber(v, 1)}
            tick={{ fontSize: 9 }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis hide />
          <RechartsTooltip
            formatter={(v: number | string | undefined, name: string | undefined) => {
              const numeric = typeof v === "number" ? v : Number(v);
              return [
                Number.isFinite(numeric) ? formatNumber(numeric, 1) : "--",
                name === "prior" ? "prior pred." : "count",
              ] as const;
            }}
            labelFormatter={(l: unknown) => {
              const numeric = typeof l === "number" ? l : Number(l);
              return Number.isFinite(numeric) ? `x = ${formatNumber(numeric, 2)}` : "x = --";
            }}
            contentStyle={{ fontSize: 10, padding: "2px 6px" }}
          />
          <Bar
            dataKey="count"
            fill="var(--muted-foreground)"
            opacity={0.3}
            barSize={isDiscrete ? 14 : undefined}
          />
          {hasPrior && (
            <Line
              type="monotone"
              dataKey="prior"
              stroke="var(--primary)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Table columns ─────────────────────────────────────────

const col = createColumnHelper<MeasurementRow>();

const columns: ColumnDef<MeasurementRow, unknown>[] = [
  col.display({
    id: "variable",
    header: "Variable",
    cell: ({ row }) => (
      <span className="font-medium font-mono text-xs">{row.original.likelihood.variable}</span>
    ),
  }),
  col.display({
    id: "distribution",
    header: "Distribution",
    cell: ({ row }) => <Badge variant="outline">{row.original.likelihood.distribution}</Badge>,
  }),
  col.display({
    id: "link",
    header: "Link",
    cell: ({ row }) => <Badge variant="secondary">{linkLabel(row.original.likelihood.link)}</Badge>,
  }),
  col.display({
    id: "chart",
    header: () => (
      <span className="inline-flex items-center gap-1">
        Data vs Prior
        <StatTooltip explanation="Empirical data histogram (grey bars) overlaid with marginal prior predictive samples (line). Compare to check whether priors imply a plausible data scale." />
      </span>
    ),
    cell: ({ row }) => <MeasurementSparkline row={row.original} />,
  }),
  col.display({
    id: "stats",
    header: "Stats",
    cell: ({ row }) => {
      const numericValues = row.original.extractions
        .map((e) => (typeof e.value === "boolean" ? (e.value ? 1 : 0) : Number(e.value)))
        .filter((v) => !Number.isNaN(v));
      if (numericValues.length === 0)
        return <span className="text-xs text-muted-foreground">--</span>;
      const mean = numericValues.reduce((s, v) => s + v, 0) / numericValues.length;
      return (
        <span className="font-mono text-xs text-muted-foreground whitespace-nowrap">
          n={numericValues.length}
          <br />
          {"μ"}={formatNumber(mean, 2)}
        </span>
      );
    },
  }),
  col.display({
    id: "reasoning",
    header: "Reasoning",
    cell: ({ row }) => (
      <span className="max-w-xs text-xs text-muted-foreground line-clamp-2">
        {row.original.likelihood.reasoning}
      </span>
    ),
  }),
];

// ── Exported component ────────────────────────────────────

export function MeasurementTable({
  likelihoods,
  extractions,
  priorPredictiveSamples,
}: {
  likelihoods: LikelihoodSpec[];
  extractions: Extraction[];
  priorPredictiveSamples?: Record<string, number[]>;
}) {
  const rows: MeasurementRow[] = likelihoods.map((lik) => ({
    likelihood: lik,
    extractions: extractions.filter((e) => e.indicator === lik.variable),
    priorSamples: priorPredictiveSamples?.[lik.variable],
  }));

  return <InfoTable columns={columns} data={rows} estimateRowHeight={88} />;
}
