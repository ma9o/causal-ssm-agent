"use client";

import { formatNumber } from "@/lib/utils/format";
import type { SVIDiagnostics } from "@causal-ssm/api-types";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ELBOLossChartProps {
  diagnostics: SVIDiagnostics;
}

export function ELBOLossChart({ diagnostics }: ELBOLossChartProps) {
  const data = diagnostics.elbo_losses.map((loss, i) => ({
    step: i + 1,
    loss,
  }));

  return (
    <div className="h-44 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 11 }}
            label={{ value: "Step", position: "insideBottom", offset: -2, fontSize: 11 }}
          />
          <YAxis
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) => formatNumber(v, 0)}
            label={{
              value: "ELBO Loss",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              fontSize: 11,
            }}
          />
          <RechartsTooltip
            formatter={
              // biome-ignore lint/suspicious/noExplicitAny: recharts overload
              ((value: number) => [formatNumber(value, 1), "ELBO"]) as any
            }
            // biome-ignore lint/suspicious/noExplicitAny: recharts overload
            labelFormatter={((label: number) => `Step ${label}`) as any}
          />
          <Line
            dataKey="loss"
            stroke="var(--primary)"
            strokeWidth={1.5}
            dot={false}
            type="monotone"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
