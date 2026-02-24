"use client";

import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { Tooltip } from "@/components/ui/tooltip";
import { evaluatePdf } from "@/lib/utils/distributions";
import { formatNumber } from "@/lib/utils/format";
import type { ParameterSpec, PriorProposal } from "@causal-ssm/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { ExternalLink } from "lucide-react";
import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

interface PriorRow extends PriorProposal {
  search_context?: string;
}

const col = createColumnHelper<PriorRow>();

/** Compact inline density chart with axes. */
function DensitySparkline({ prior }: { prior: PriorRow }) {
  const data =
    prior.density_points ??
    evaluatePdf(prior.distribution, prior.params as Record<string, number>, 60);
  return (
    <div className="h-16 w-36">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 2, right: 4, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="x"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 9 }}
            tickFormatter={(v: number) => formatNumber(v, 1)}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis hide />
          <RechartsTooltip
            formatter={(v: number | string | undefined) => {
              const numeric = typeof v === "number" ? v : Number(v);
              return [
                Number.isFinite(numeric) ? formatNumber(numeric, 4) : "--",
                "density",
              ] as const;
            }}
            labelFormatter={(l: unknown) => {
              const numeric = typeof l === "number" ? l : Number(l);
              return Number.isFinite(numeric) ? `x = ${formatNumber(numeric, 3)}` : "x = --";
            }}
            contentStyle={{ fontSize: 10, padding: "2px 6px" }}
          />
          <Area
            type="monotone"
            dataKey="y"
            stroke="var(--primary)"
            fill="var(--primary)"
            fillOpacity={0.15}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function formatParams(params: Record<string, number>): string {
  return Object.entries(params)
    .map(([k, v]) => `${k}=${formatNumber(v, 2)}`)
    .join(", ");
}

const baseColumns = [
  col.accessor("parameter", {
    header: "Parameter",
    cell: (info) => <span className="font-medium font-mono text-xs">{info.getValue()}</span>,
  }),
  col.accessor("distribution", {
    header: "Distribution",
    cell: (info) => <Badge variant="outline">{info.getValue()}</Badge>,
  }),
  col.display({
    id: "params",
    header: "Params",
    cell: ({ row }) => (
      <span className="font-mono text-xs text-muted-foreground">
        {formatParams(row.original.params as Record<string, number>)}
      </span>
    ),
  }),
  col.display({
    id: "density",
    header: "Density",
    cell: ({ row }) => <DensitySparkline prior={row.original} />,
  }),
  col.accessor("reasoning", {
    header: "Reasoning",
    cell: (info) => (
      <span className="max-w-xs text-xs text-muted-foreground line-clamp-2">{info.getValue()}</span>
    ),
  }),
  col.display({
    id: "sources",
    header: () => (
      <HeaderWithTooltip
        label="Sources"
        tooltip="Literature sources supporting this prior choice. Click to open."
      />
    ),
    cell: ({ row }) => {
      const sources = row.original.sources;
      if (sources.length === 0) {
        return <span className="text-xs text-muted-foreground">--</span>;
      }
      return (
        <div className="flex items-center gap-1.5">
          {sources.map((source, i) => (
            <Tooltip
              key={`source-${
                // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                i
              }`}
              content={
                <div className="max-w-xs text-xs">
                  <p className="font-medium">{source.title}</p>
                  <p className="text-muted-foreground">{source.snippet}</p>
                  {source.effect_size && (
                    <span className="text-muted-foreground">Effect: {source.effect_size}</span>
                  )}
                </div>
              }
            >
              {source.url ? (
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-0.5 text-primary hover:underline"
                >
                  <Badge variant="secondary" className="cursor-pointer text-[10px] px-1.5">
                    {i + 1}
                    <ExternalLink className="ml-0.5 h-2.5 w-2.5" />
                  </Badge>
                </a>
              ) : (
                <Badge variant="secondary" className="text-[10px] px-1.5">
                  {i + 1}
                </Badge>
              )}
            </Tooltip>
          ))}
        </div>
      );
    },
    meta: { align: "center" },
  }),
] as ColumnDef<PriorRow, unknown>[];

const searchContextColumn = col.accessor("search_context", {
  header: () => (
    <HeaderWithTooltip
      label="Search Context"
      tooltip="The search query used by the pipeline to find prior literature for this parameter's effect size."
    />
  ),
  cell: (info) => (
    <span className="max-w-xs text-xs text-muted-foreground italic">{info.getValue() || "--"}</span>
  ),
}) as ColumnDef<PriorRow, unknown>;

export function PriorTable({
  priors,
  parameters,
}: { priors: PriorProposal[]; parameters?: ParameterSpec[] }) {
  const paramMap = useMemo(() => {
    const map = new Map<string, ParameterSpec>();
    for (const p of parameters ?? []) map.set(p.name, p);
    return map;
  }, [parameters]);

  const rows: PriorRow[] = useMemo(
    () => priors.map((p) => ({ ...p, search_context: paramMap.get(p.parameter)?.search_context })),
    [priors, paramMap],
  );

  const hasSearchContext = rows.some((r) => r.search_context);

  const columns = useMemo<ColumnDef<PriorRow, unknown>[]>(
    () => (hasSearchContext ? [...baseColumns, searchContextColumn] : baseColumns),
    [hasSearchContext],
  );

  return <InfoTable columns={columns} data={rows} estimateRowHeight={72} />;
}
