"use client";

import { formatNumber } from "@/lib/utils/format";
import type { PosteriorPair } from "@causal-ssm/api-types";
import {
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from "recharts";

interface PosteriorPairsChartProps {
  pair: PosteriorPair;
}

export function PosteriorPairsChart({ pair }: PosteriorPairsChartProps) {
  const divFlags = pair.divergent ?? [];
  const hasDivergent = divFlags.some(Boolean);

  const normal: { x: number; y: number }[] = [];
  const divergent: { x: number; y: number }[] = [];

  for (let i = 0; i < pair.x_values.length; i++) {
    const point = { x: pair.x_values[i], y: pair.y_values[i] };
    if (hasDivergent && divFlags[i]) {
      divergent.push(point);
    } else {
      normal.push(point);
    }
  }

  return (
    <div className="space-y-1">
      <span className="text-xs font-mono text-muted-foreground">
        {pair.param_x} vs {pair.param_y}
        {hasDivergent && (
          <span className="ml-2 text-destructive">({divergent.length} divergent)</span>
        )}
      </span>
      <div className="h-36 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="x"
              type="number"
              tick={{ fontSize: 9 }}
              tickFormatter={(v: number) => formatNumber(v, 2)}
              label={{ value: pair.param_x, position: "insideBottom", offset: -2, fontSize: 9 }}
            />
            <YAxis
              dataKey="y"
              type="number"
              tick={{ fontSize: 9 }}
              tickFormatter={(v: number) => formatNumber(v, 2)}
              label={{
                value: pair.param_y,
                angle: -90,
                position: "insideLeft",
                offset: 10,
                fontSize: 9,
              }}
            />
            <RechartsTooltip formatter={(value) => [formatNumber(Number(value), 3)]} />
            <Scatter data={normal} fill="var(--primary)" fillOpacity={0.3} r={2} name="normal" />
            {hasDivergent && (
              <Scatter
                data={divergent}
                fill="var(--destructive)"
                fillOpacity={0.8}
                r={3}
                name="divergent"
              />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
