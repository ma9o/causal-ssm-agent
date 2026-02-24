"use client";

import { CHAIN_COLORS } from "@/lib/constants/charts";
import type { RankHistogram as RankHistogramType } from "@causal-ssm/api-types";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";


interface RankHistogramProps {
  histogram: RankHistogramType;
}

export function RankHistogram({ histogram }: RankHistogramProps) {
  // Build data: [{bin, chain_0, chain_1, ...}]
  const nBins = histogram.n_bins;
  const data = Array.from({ length: nBins }, (_, i) => {
    const row: Record<string, number> = { bin: i + 1 };
    for (const ch of histogram.chains) {
      row[`chain_${ch.chain}`] = ch.counts[i];
    }
    return row;
  });

  return (
    <div className="space-y-1">
      <span className="text-xs font-mono text-muted-foreground">{histogram.parameter}</span>
      <div className="h-28 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" vertical={false} />
            <XAxis dataKey="bin" tick={{ fontSize: 9 }} />
            <YAxis tick={{ fontSize: 9 }} hide />
            <RechartsTooltip
              formatter={
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                ((value: number, name: string) => [value, name]) as any
              }
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
                fill={CHAIN_COLORS[ch.chain % CHAIN_COLORS.length]}
                fillOpacity={0.5}
                name={`Chain ${ch.chain}`}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
