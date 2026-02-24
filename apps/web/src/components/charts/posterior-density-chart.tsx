"use client";

import { Badge } from "@/components/ui/badge";
import { formatNumber } from "@/lib/utils/format";
import type { PosteriorMarginal } from "@causal-ssm/api-types";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

interface PosteriorDensityChartProps {
  marginal: PosteriorMarginal;
}

export function PosteriorDensityChart({ marginal }: PosteriorDensityChartProps) {
  const data = marginal.x_values.map((x, i) => ({
    x,
    density: marginal.density[i],
  }));

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-muted-foreground">{marginal.parameter}</span>
        <Badge variant="outline" className="text-[10px]">
          {formatNumber(marginal.mean, 3)} +/- {formatNumber(marginal.sd, 3)}
        </Badge>
      </div>
      <div className="h-28 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="x"
              type="number"
              tick={{ fontSize: 10 }}
              tickFormatter={(v: number) => formatNumber(v, 2)}
            />
            <YAxis tick={{ fontSize: 10 }} hide />
            <RechartsTooltip
              formatter={
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                ((value: number) => [formatNumber(value, 3), "Density"]) as any
              }
              // biome-ignore lint/suspicious/noExplicitAny: recharts overload
              labelFormatter={((label: number) => formatNumber(label, 3)) as any}
            />
            {/* HDI shading would need a more complex approach; use reference lines */}
            <ReferenceLine
              x={marginal.hdi_3}
              stroke="var(--muted-foreground)"
              strokeDasharray="4 4"
              strokeWidth={1}
            />
            <ReferenceLine
              x={marginal.hdi_97}
              stroke="var(--muted-foreground)"
              strokeDasharray="4 4"
              strokeWidth={1}
            />
            <ReferenceLine
              x={marginal.mean}
              stroke="var(--foreground)"
              strokeWidth={1.5}
            />
            <Area
              dataKey="density"
              stroke="var(--primary)"
              fill="var(--primary)"
              fillOpacity={0.2}
              strokeWidth={1.5}
              type="monotone"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
