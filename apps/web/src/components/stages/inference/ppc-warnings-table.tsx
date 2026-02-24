"use client";

import { HeaderWithTooltip } from "@/components/ui/info-table";
import { InfoTable } from "@/components/ui/info-table";
import { formatNumber } from "@/lib/utils/format";
import { buildHistogram } from "@/lib/utils/histogram";
import type { PPCWarning, PPCTestStat, PPCOverlay } from "@causal-ssm/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

// ── Row type (one per variable) ──────────────────────────

type CheckType = "calibration" | "autocorrelation" | "variance";
type StatName = "mean" | "sd" | "min" | "max";

interface PPCVariableRow {
  variable: string;
  checks: Partial<Record<CheckType, PPCWarning>>;
  testStats: Partial<Record<StatName, PPCTestStat>>;
  overlay?: PPCOverlay;
}

function buildRows(
  warnings: PPCWarning[],
  testStats: PPCTestStat[],
  overlays: PPCOverlay[],
): PPCVariableRow[] {
  const map = new Map<string, PPCVariableRow>();
  for (const w of warnings) {
    if (!map.has(w.variable))
      map.set(w.variable, { variable: w.variable, checks: {}, testStats: {} });
    map.get(w.variable)!.checks[w.check_type] = w;
  }
  for (const ts of testStats) {
    if (!map.has(ts.variable))
      map.set(ts.variable, { variable: ts.variable, checks: {}, testStats: {} });
    map.get(ts.variable)!.testStats[ts.stat_name as StatName] = ts;
  }
  for (const ov of overlays) {
    if (!map.has(ov.variable))
      map.set(ov.variable, { variable: ov.variable, checks: {}, testStats: {} });
    map.get(ov.variable)!.overlay = ov;
  }
  return Array.from(map.values());
}

// ── Test stat sparkline (mini histogram + p-value) ──────

function TestStatSparkline({ stat }: { stat?: PPCTestStat }) {
  if (!stat)
    return <span className="text-xs text-muted-foreground">—</span>;

  const bins = buildHistogram(stat.rep_values, 12);
  const pValue =
    stat.rep_values.filter((v) => v >= stat.observed_value).length /
    stat.rep_values.length;

  return (
    <div className="space-y-1">
      <span className="text-xs font-mono">p = {formatNumber(pValue, 2)}</span>
      <div className="h-14 w-28">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={bins}
            margin={{ top: 2, right: 2, left: 0, bottom: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              className="stroke-muted"
              vertical={false}
            />
            <XAxis
              dataKey="binCenter"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={false}
              axisLine={{ stroke: "var(--border)" }}
              height={2}
            />
            <YAxis hide />
            <Bar dataKey="count" fill="var(--primary)" fillOpacity={0.5} />
            <ReferenceLine
              x={stat.observed_value}
              stroke="var(--foreground)"
              strokeWidth={1.5}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ── Overlay sparkline (mini ribbon chart) ────────────────

function OverlaySparkline({ overlay }: { overlay?: PPCOverlay }) {
  if (!overlay)
    return <span className="text-xs text-muted-foreground">—</span>;

  const data = overlay.observed.map((obs, i) => ({
    t: i,
    observed: obs,
    q025: overlay.q025[i],
    q975: overlay.q975[i],
    median: overlay.median[i],
  }));

  return (
    <div className="h-16 w-48">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={data}
          margin={{ top: 2, right: 2, left: 0, bottom: 0 }}
        >
          <YAxis hide />
          <XAxis dataKey="t" hide />
          <Area
            dataKey="q975"
            stroke="none"
            fill="var(--primary)"
            fillOpacity={0.15}
            type="monotone"
            isAnimationActive={false}
          />
          <Area
            dataKey="q025"
            stroke="none"
            fill="var(--background)"
            fillOpacity={1}
            type="monotone"
            isAnimationActive={false}
          />
          <Line
            dataKey="median"
            stroke="var(--primary)"
            strokeWidth={1}
            strokeDasharray="3 3"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            dataKey="observed"
            stroke="var(--foreground)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Table columns ────────────────────────────────────────

const col = createColumnHelper<PPCVariableRow>();

const CHECK_TYPES: CheckType[] = ["calibration", "autocorrelation", "variance"];
const STAT_NAMES: StatName[] = ["mean", "sd", "min", "max"];

const CHECK_TOOLTIPS: Record<CheckType, string> = {
  calibration:
    "Fraction of observed timepoints falling within the 95% posterior predictive interval. Expected ~0.95.",
  autocorrelation:
    "Lag-1 autocorrelation of residuals (observed minus predicted mean). High values suggest missing dynamics.",
  variance:
    "Ratio of posterior predictive std to observed std. Values far from 1 indicate scale misfit.",
};

const STAT_TOOLTIPS: Record<StatName, string> = {
  mean: "Compares observed mean to distribution of replicated means. Vertical line is T(y).",
  sd: "Compares observed std dev to distribution of replicated std devs.",
  min: "Compares observed minimum to distribution of replicated minima.",
  max: "Compares observed maximum to distribution of replicated maxima.",
};

function pValueForStat(stat: PPCTestStat): number {
  return stat.rep_values.filter((v) => v >= stat.observed_value).length / stat.rep_values.length;
}

const columns: ColumnDef<PPCVariableRow, unknown>[] = [
  col.display({
    id: "variable",
    header: "Variable",
    cell: ({ row }) => (
      <span className="font-medium font-mono text-xs">
        {row.original.variable}
      </span>
    ),
  }),
  col.display({
    id: "overlay",
    header: () => (
      <HeaderWithTooltip
        label="y vs y_rep"
        tooltip="Observed data (solid) vs 95% posterior predictive band (shaded) and median (dashed). Data outside the band suggests misfit."
      />
    ),
    cell: ({ row }) => <OverlaySparkline overlay={row.original.overlay} />,
  }),
  ...CHECK_TYPES.map((ct) =>
    col.display({
      id: ct,
      header: () => (
        <HeaderWithTooltip
          label={ct === "autocorrelation" ? "Autocorr" : ct.charAt(0).toUpperCase() + ct.slice(1)}
          tooltip={CHECK_TOOLTIPS[ct]}
        />
      ),
      cell: ({ row }) => {
        const warning = row.original.checks[ct];
        if (!warning) return <span className="text-xs text-muted-foreground">—</span>;
        return (
          <span className="font-mono text-xs" title={warning.message}>
            {formatNumber(warning.value)}
          </span>
        );
      },
      meta: {
        severity: (_v: unknown, row: PPCVariableRow) => {
          const warning = row.checks[ct];
          return warning && !warning.passed ? "fail" : undefined;
        },
      },
    }),
  ),
  ...STAT_NAMES.map((sn) =>
    col.display({
      id: `t_${sn}`,
      header: () => (
        <HeaderWithTooltip
          label={`T(${sn})`}
          tooltip={STAT_TOOLTIPS[sn]}
        />
      ),
      cell: ({ row }) => <TestStatSparkline stat={row.original.testStats[sn]} />,
      meta: {
        severity: (_v: unknown, row: PPCVariableRow) => {
          const stat = row.testStats[sn];
          if (!stat) return undefined;
          const p = pValueForStat(stat);
          return (p < 0.05 || p > 0.95) ? "fail" : undefined;
        },
      },
    }),
  ),
];

// ── Exported component ───────────────────────────────────

export function PPCWarningsTable({
  warnings,
  testStats,
  overlays,
}: {
  warnings: PPCWarning[];
  testStats: PPCTestStat[];
  overlays: PPCOverlay[];
}) {
  const rows = buildRows(warnings, testStats, overlays);
  if (rows.length === 0) return null;

  return (
    <InfoTable
      columns={columns}
      data={rows}
      estimateRowHeight={80}
      sorting={false}
    />
  );
}
