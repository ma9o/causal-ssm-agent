"use client";

import { formatNumber } from "@/lib/utils/format";
import type { LOODiagnostics } from "@causal-ssm/api-types";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ParetoKChartProps {
  loo: LOODiagnostics;
}

export function ParetoKChart({ loo }: ParetoKChartProps) {
  if (!loo.pareto_k || loo.pareto_k.length === 0) return null;

  const data = loo.pareto_k.map((k, i) => ({
    timestep: i + 1,
    k,
  }));

  return (
    <div className="space-y-2">
      <span className="text-xs font-mono text-muted-foreground">Pareto k per Timestep</span>
      <div className="h-44 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="timestep"
              type="number"
              tick={{ fontSize: 10 }}
              label={{ value: "Timestep", position: "insideBottom", offset: -2, fontSize: 10 }}
            />
            <YAxis
              dataKey="k"
              tick={{ fontSize: 10 }}
              label={{ value: "Pareto k", angle: -90, position: "insideLeft", offset: 10, fontSize: 10 }}
            />
            <RechartsTooltip
              formatter={
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                ((value: number) => [formatNumber(value, 3), "Pareto k"]) as any
              }
            />
            {/* Threshold lines */}
            <ReferenceLine
              y={0.7}
              stroke="var(--destructive)"
              strokeDasharray="4 4"
              label={{ value: "k = 0.7", position: "right", fontSize: 9, fill: "var(--destructive)" }}
            />
            <ReferenceLine
              y={0.5}
              stroke="var(--warning)"
              strokeDasharray="4 4"
              label={{ value: "k = 0.5", position: "right", fontSize: 9, fill: "var(--warning)" }}
            />
            <Scatter
              data={data}
              fill="var(--primary)"
              fillOpacity={0.6}
              r={3}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">
        Pareto k diagnostic per timestep (one-step-ahead predictive). Values above 0.7 indicate
        the timestep is highly influential and LOO estimate may be unreliable.
      </p>
    </div>
  );
}
