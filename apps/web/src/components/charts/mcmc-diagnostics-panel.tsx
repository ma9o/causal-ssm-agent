"use client";

import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { CHAIN_COLORS } from "@/lib/constants/charts";
import {
  DEFAULT_N_SAMPLES,
  ESS_RATIO_FAIL,
  ESS_RATIO_WARN,
  RHAT_FAIL,
  RHAT_WARN,
} from "@/lib/constants/diagnostics";
import { formatNumber } from "@/lib/utils/format";
import type {
  MCMCDiagnostics,
  MCMCParamDiagnostic,
  RankHistogram as RankHistogramData,
  TraceData,
} from "@causal-ssm/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { AlertTriangle } from "lucide-react";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Line,
  LineChart,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface MCMCDiagnosticsPanelProps {
  diagnostics: MCMCDiagnostics;
}

interface EnrichedParamRow extends MCMCParamDiagnostic {
  trace?: TraceData;
  rank?: RankHistogramData;
}

/* ── Severity helpers ── */

function rhatSeverity(value: number | number[]): "fail" | "warn" | undefined {
  const v = Array.isArray(value) ? Math.max(...value) : value;
  if (v >= RHAT_FAIL) return "fail";
  if (v >= RHAT_WARN) return "warn";
  return undefined;
}

function essSeverity(
  value: number | number[] | undefined,
  nSamples: number | null,
): "fail" | "warn" | undefined {
  if (value == null) return undefined;
  const v = Array.isArray(value) ? Math.min(...value) : value;
  const total = nSamples ?? DEFAULT_N_SAMPLES;
  const ratio = v / total;
  if (ratio <= ESS_RATIO_FAIL) return "fail";
  if (ratio <= ESS_RATIO_WARN) return "warn";
  return undefined;
}

/* ── Inline recharts sparklines ── */

function InlineTrace({ trace }: { trace: TraceData }) {
  const nPoints = trace.chains[0]?.values.length ?? 0;
  const data = Array.from({ length: nPoints }, (_, i) => {
    const row: Record<string, number> = { draw: i };
    for (const ch of trace.chains) {
      row[`chain_${ch.chain}`] = ch.values[i];
    }
    return row;
  });

  return (
    <div className="h-11 w-[200px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
          <RechartsTooltip
            formatter={(value, name) => [formatNumber(Number(value), 3), String(name)]}
            contentStyle={{ fontSize: 11, padding: "4px 8px", background: "white", zIndex: 50 }}
            wrapperStyle={{ zIndex: 50 }}
          />
          {trace.chains.map((ch) => (
            <Line
              key={ch.chain}
              dataKey={`chain_${ch.chain}`}
              style={{ stroke: CHAIN_COLORS[ch.chain % CHAIN_COLORS.length] }}
              strokeWidth={1}
              dot={false}
              name={`Chain ${ch.chain}`}
              opacity={0.7}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function InlineRankHist({ histogram }: { histogram: RankHistogramData }) {
  const data = Array.from({ length: histogram.n_bins }, (_, i) => {
    const row: Record<string, number> = { bin: i + 1 };
    for (const ch of histogram.chains) {
      row[`chain_${ch.chain}`] = ch.counts[i];
    }
    return row;
  });

  return (
    <div className="h-11 w-[160px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
          <RechartsTooltip
            formatter={(value, name) => [String(value), String(name)]}
            contentStyle={{ fontSize: 11, padding: "4px 8px", background: "white", zIndex: 50 }}
            wrapperStyle={{ zIndex: 50 }}
          />
          <ReferenceLine
            y={histogram.expected_per_bin}
            stroke="var(--muted-foreground)"
            strokeDasharray="4 4"
            strokeWidth={1}
          />
          {histogram.chains.map((ch) => (
            <Bar
              key={ch.chain}
              dataKey={`chain_${ch.chain}`}
              style={{ fill: CHAIN_COLORS[ch.chain % CHAIN_COLORS.length] }}
              fillOpacity={0.5}
              name={`Chain ${ch.chain}`}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Column helper ── */

const col = createColumnHelper<EnrichedParamRow>();

export function MCMCDiagnosticsPanel({ diagnostics }: MCMCDiagnosticsPanelProps) {
  const hasDivergences = diagnostics.num_divergences > 0;
  const hasEssTail = diagnostics.per_parameter.some((p) => p.ess_tail != null);
  const hasMcse = diagnostics.per_parameter.some((p) => p.mcse_mean != null);
  const hasTraces = (diagnostics.trace_data?.length ?? 0) > 0;
  const hasRankHists = (diagnostics.rank_histograms?.length ?? 0) > 0;

  const enrichedData = useMemo<EnrichedParamRow[]>(() => {
    const traceByParam = new Map((diagnostics.trace_data ?? []).map((t) => [t.parameter, t]));
    const rankByParam = new Map((diagnostics.rank_histograms ?? []).map((h) => [h.parameter, h]));
    return diagnostics.per_parameter.map((p) => ({
      ...p,
      trace: traceByParam.get(p.parameter),
      rank: rankByParam.get(p.parameter),
    }));
  }, [diagnostics.per_parameter, diagnostics.trace_data, diagnostics.rank_histograms]);

  const columns = useMemo<ColumnDef<EnrichedParamRow, unknown>[]>(() => {
    const cols = [
      col.accessor("parameter", {
        header: "Parameter",
        cell: (info) => <span className="font-medium">{info.getValue()}</span>,
        meta: { mono: true },
      }),
      ...(hasTraces
        ? [
            col.display({
              id: "trace",
              header: () => (
                <HeaderWithTooltip
                  label="Trace"
                  tooltip="Per-chain trace plot. Chains should mix well (look like a 'hairy caterpillar')."
                />
              ),
              cell: (info) => {
                const t = info.row.original.trace;
                return t ? (
                  <InlineTrace trace={t} />
                ) : (
                  <span className="text-muted-foreground">—</span>
                );
              },
              enableSorting: false,
            }) as ColumnDef<EnrichedParamRow, unknown>,
          ]
        : []),
      ...(hasRankHists
        ? [
            col.display({
              id: "rank",
              header: () => (
                <HeaderWithTooltip
                  label="Rank hist."
                  tooltip="Rank histogram for chain mixing. Uniform bars indicate good mixing across chains."
                />
              ),
              cell: (info) => {
                const r = info.row.original.rank;
                return r ? (
                  <InlineRankHist histogram={r} />
                ) : (
                  <span className="text-muted-foreground">—</span>
                );
              },
              enableSorting: false,
            }) as ColumnDef<EnrichedParamRow, unknown>,
          ]
        : []),
      col.accessor("r_hat", {
        header: () => (
          <HeaderWithTooltip
            label="R-hat"
            tooltip="Potential scale reduction factor. Values near 1.0 indicate convergence. Worry above 1.01."
          />
        ),
        cell: (info) => {
          const v = info.getValue();
          const val = Array.isArray(v) ? Math.max(...v) : v;
          return formatNumber(val, 3);
        },
        meta: {
          align: "right",
          mono: true,
          severity: (v: number | number[]) => rhatSeverity(v),
        },
      }),
      col.accessor("ess_bulk", {
        header: () => (
          <HeaderWithTooltip
            label="ESS (bulk)"
            tooltip="Effective sample size for bulk of the distribution. Higher is better. Worry if < 100 per chain."
          />
        ),
        cell: (info) => {
          const v = info.getValue();
          if (v == null) return <span className="text-muted-foreground">—</span>;
          const val = Array.isArray(v) ? Math.min(...v) : v;
          return formatNumber(val, 0);
        },
        meta: {
          align: "right",
          mono: true,
          severity: (v: number | number[] | undefined) =>
            essSeverity(v, diagnostics.num_samples ?? null),
        },
      }),
    ] as ColumnDef<EnrichedParamRow, unknown>[];

    if (hasEssTail) {
      cols.push(
        col.accessor("ess_tail", {
          header: () => (
            <HeaderWithTooltip
              label="ESS (tail)"
              tooltip="Effective sample size for the tails (5th/95th percentiles). Important for credible interval reliability."
            />
          ),
          cell: (info) => {
            const v = info.getValue();
            if (v == null) return <span className="text-muted-foreground">—</span>;
            const val = Array.isArray(v) ? Math.min(...v) : v;
            return formatNumber(val, 0);
          },
          meta: {
            align: "right",
            mono: true,
            severity: (v: number | number[] | null | undefined) =>
              essSeverity(v ?? undefined, diagnostics.num_samples ?? null),
          },
        }) as ColumnDef<EnrichedParamRow, unknown>,
      );
    }

    if (hasMcse) {
      cols.push(
        col.accessor("mcse_mean", {
          header: () => (
            <HeaderWithTooltip
              label="MCSE"
              tooltip="Monte Carlo standard error of the mean. Should be small relative to the posterior standard deviation."
            />
          ),
          cell: (info) => {
            const v = info.getValue();
            if (v == null) return <span className="text-muted-foreground">—</span>;
            const val = Array.isArray(v) ? Math.max(...v) : v;
            return formatNumber(val, 4);
          },
          meta: { align: "right", mono: true },
        }) as ColumnDef<EnrichedParamRow, unknown>,
      );
    }

    return cols;
  }, [hasEssTail, hasMcse, hasTraces, hasRankHists, diagnostics.num_samples]);

  return (
    <div className="space-y-3">
      {/* Sampler-level summary */}
      <div className="flex flex-wrap gap-2">
        {diagnostics.num_chains != null && (
          <Badge variant="secondary">{diagnostics.num_chains} chains</Badge>
        )}
        {diagnostics.num_samples != null && (
          <Badge variant="secondary">{diagnostics.num_samples.toLocaleString()} samples</Badge>
        )}
        <Badge variant={hasDivergences ? "destructive" : "success"}>
          {hasDivergences && <AlertTriangle className="mr-1 h-3 w-3" />}
          {diagnostics.num_divergences} divergence{diagnostics.num_divergences !== 1 && "s"}
          {hasDivergences && ` (${formatNumber(diagnostics.divergence_rate * 100, 1)}%)`}
        </Badge>
        <Badge variant="secondary">
          tree depth: {formatNumber(diagnostics.tree_depth_mean, 1)} avg,{" "}
          {diagnostics.tree_depth_max} max
        </Badge>
        <Badge variant="secondary">
          accept: {formatNumber(diagnostics.accept_prob_mean * 100, 1)}%
        </Badge>
      </div>

      {/* Per-parameter table with inline traces & rank histograms */}
      {enrichedData.length > 0 && (
        <InfoTable columns={columns} data={enrichedData} estimateRowHeight={60} />
      )}
    </div>
  );
}
