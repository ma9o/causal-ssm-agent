"use client";

import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import { formatNumber } from "@/lib/utils/format";
import type { ParameterIdentification } from "@causal-ssm/api-types";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

function ProfileSparkline({
  param,
  threshold,
}: {
  param: ParameterIdentification;
  threshold: number;
}) {
  if (!param.profile_x || !param.profile_ll) return <span className="text-xs text-muted-foreground">—</span>;

  const data = param.profile_x.map((x, i) => ({
    x,
    ll: param.profile_ll![i],
  }));

  return (
    <div className="h-16 w-36">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 4, left: 0, bottom: 0 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            className="stroke-muted"
            vertical={false}
          />
          <XAxis
            dataKey="x"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 9 }}
            tickFormatter={(v: number) => formatNumber(v, 1)}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis hide domain={["dataMin", "auto"]} />
          <RechartsTooltip
            formatter={(v: number | string | undefined) => {
              const numeric = typeof v === "number" ? v : Number(v);
              return [
                Number.isFinite(numeric) ? formatNumber(numeric, 2) : "--",
                "profile \u2113\u2113",
              ] as const;
            }}
            labelFormatter={(l: unknown) => {
              const numeric = typeof l === "number" ? l : Number(l);
              return Number.isFinite(numeric) ? `\u03B8 = ${formatNumber(numeric, 3)}` : "\u03B8 = --";
            }}
            contentStyle={{ fontSize: 10, padding: "2px 6px" }}
          />
          <ReferenceLine
            y={-threshold}
            stroke="var(--muted-foreground)"
            strokeDasharray="3 3"
            strokeWidth={1}
          />
          <Line
            type="monotone"
            dataKey="ll"
            stroke="var(--primary)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const col = createColumnHelper<ParameterIdentification>();

function buildColumns(threshold: number) {
  return [
    col.accessor("name", {
      header: "Parameter",
      cell: (info) => (
        <span className="font-medium">{info.getValue()}</span>
      ),
      meta: { mono: true },
    }),
    col.accessor("contraction_ratio", {
      header: () => (
        <HeaderWithTooltip
          label="Contraction"
          tooltip="Prior-to-posterior contraction ratio. Values near 1 mean the data strongly informs the parameter; values near 0 mean the posterior is dominated by the prior (weak identification)."
        />
      ),
      cell: (info) => {
        const v = info.getValue();
        return v != null ? formatNumber(v) : "--";
      },
      meta: {
        mono: true,
        severity: (_v: number | null, row: ParameterIdentification) => {
          if (row.classification === "structurally_unidentifiable") return "fail";
          if (row.classification === "practically_unidentifiable") return "warn";
          return undefined;
        },
      },
    }),
    col.display({
      id: "profile",
      header: () => (
        <HeaderWithTooltip
          label="Profile Likelihood"
          tooltip="Profile likelihood curve: shows how log-likelihood changes as each parameter is varied with others optimized out. The dashed line is the chi-squared threshold — if the curve stays above it, the parameter is unidentifiable."
        />
      ),
      cell: (info) => (
        <ProfileSparkline param={info.row.original} threshold={threshold} />
      ),
    }),
  ];
}

export function WeakParamsList({
  params,
  threshold,
}: {
  params: ParameterIdentification[];
  threshold?: number | null;
}) {
  const columns = buildColumns(threshold ?? 1.92);
  return <InfoTable columns={columns as ColumnDef<ParameterIdentification, unknown>[]} data={params} estimateRowHeight={80} />;
}
