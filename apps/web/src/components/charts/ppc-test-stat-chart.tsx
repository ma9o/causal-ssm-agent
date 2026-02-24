"use client";

import { Badge } from "@/components/ui/badge";
import { formatNumber } from "@/lib/utils/format";
import { buildHistogram } from "@/lib/utils/histogram";
import type { PPCTestStat } from "@causal-ssm/api-types";
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

interface PPCTestStatChartProps {
  stat: PPCTestStat;
}


export function PPCTestStatChart({ stat }: PPCTestStatChartProps) {
  const bins = buildHistogram(stat.rep_values);
  const pValue = stat.rep_values.filter((v) => v >= stat.observed_value).length / stat.rep_values.length;
  const isExtreme = pValue < 0.05 || pValue > 0.95;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-muted-foreground">
          T = {stat.stat_name}({stat.variable})
        </span>
        <Badge variant={isExtreme ? "destructive" : "success"} className="text-[10px]">
          p = {formatNumber(pValue, 2)}
        </Badge>
      </div>
      <div className="h-36 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" vertical={false} />
            <XAxis
              dataKey="binCenter"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fontSize: 10 }}
              tickFormatter={(v: number) => formatNumber(v, 1)}
            />
            <YAxis tick={{ fontSize: 10 }} hide />
            <RechartsTooltip
              formatter={
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                ((value: number) => [value, "Count"]) as any
              }
              // biome-ignore lint/suspicious/noExplicitAny: recharts overload
              labelFormatter={
                ((label: number) => `T(y_rep) \u2248 ${formatNumber(label, 2)}`) as any
              }
            />
            <Bar dataKey="count" fill="var(--primary)" fillOpacity={0.5} />
            <ReferenceLine
              x={stat.observed_value}
              stroke="var(--foreground)"
              strokeWidth={2}
              strokeDasharray="0"
              label={{
                value: `T(y) = ${formatNumber(stat.observed_value, 2)}`,
                position: "top",
                fontSize: 10,
                fill: "var(--foreground)",
              }}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
