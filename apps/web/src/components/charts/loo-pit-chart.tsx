"use client";

import { formatNumber } from "@/lib/utils/format";
import type { LOODiagnostics } from "@causal-ssm/api-types";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

interface LOOPITChartProps {
  loo: LOODiagnostics;
}

export function LOOPITChart({ loo }: LOOPITChartProps) {
  if (!loo.loo_pit || loo.loo_pit.length === 0) return null;

  // Build empirical CDF of LOO-PIT values
  const sorted = [...loo.loo_pit].sort((a, b) => a - b);
  const n = sorted.length;
  const data = sorted.map((v, i) => ({
    pit: v,
    ecdf: (i + 1) / n,
    uniform: v, // 45-degree line
  }));

  return (
    <div className="space-y-2">
      <span className="text-xs font-mono text-muted-foreground">LOO-PIT Calibration</span>
      <div className="h-44 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="pit"
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 11 }}
              label={{ value: "LOO-PIT", position: "insideBottom", offset: -2, fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 11 }}
              label={{ value: "ECDF", angle: -90, position: "insideLeft", offset: 10, fontSize: 11 }}
            />
            <RechartsTooltip
              formatter={
                // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                ((value: number, name: string) => {
                  const labels: Record<string, string> = {
                    ecdf: "ECDF",
                    uniform: "Uniform",
                  };
                  return [formatNumber(value, 3), labels[name] ?? name];
                }) as any
              }
            />
            {/* Reference: uniform CDF (45-degree line) */}
            <Line
              dataKey="uniform"
              stroke="var(--muted-foreground)"
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
              name="uniform"
            />
            {/* Empirical CDF of LOO-PIT values */}
            <Line
              dataKey="ecdf"
              stroke="var(--primary)"
              strokeWidth={2}
              dot={false}
              name="ecdf"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">
        LOO-PIT: If the model is well-calibrated, the ECDF should follow the diagonal (uniform).
        Deviations indicate miscalibration.
      </p>
    </div>
  );
}
