"use client";

import { CHAIN_COLORS } from "@/lib/constants/charts";
import { formatNumber } from "@/lib/utils/format";
import type { TraceData } from "@causal-ssm/api-types";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";


interface TracePlotProps {
  trace: TraceData;
}

export function TracePlot({ trace }: TracePlotProps) {
  // Build data array: [{draw, chain_0, chain_1, ...}]
  const nPoints = trace.chains[0]?.values.length ?? 0;
  const data = Array.from({ length: nPoints }, (_, i) => {
    const row: Record<string, number> = { draw: i };
    for (const ch of trace.chains) {
      row[`chain_${ch.chain}`] = ch.values[i];
    }
    return row;
  });

  return (
    <div className="space-y-1">
      <span className="text-xs font-mono text-muted-foreground">{trace.parameter}</span>
      <div className="h-32 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="draw"
              tick={{ fontSize: 10 }}
              label={{ value: "Draw", position: "insideBottom", offset: -2, fontSize: 10 }}
            />
            <YAxis tick={{ fontSize: 10 }} tickFormatter={(v: number) => formatNumber(v, 2)} />
            <RechartsTooltip
              formatter={
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                ((value: number, name: string) => [formatNumber(value, 3), name]) as any
              }
            />
            {trace.chains.map((ch) => (
              <Line
                key={ch.chain}
                dataKey={`chain_${ch.chain}`}
                stroke={CHAIN_COLORS[ch.chain % CHAIN_COLORS.length]}
                strokeWidth={1}
                dot={false}
                name={`Chain ${ch.chain}`}
                opacity={0.7}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
