"use client";

import { formatNumber } from "@/lib/utils/format";
import type { PPCOverlay } from "@causal-ssm/api-types";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

interface PPCRibbonChartProps {
  overlay: PPCOverlay;
}

export function PPCRibbonChart({ overlay }: PPCRibbonChartProps) {
  const hasSpaghetti = overlay.spaghetti_draws && overlay.spaghetti_draws.length > 0;

  const data = overlay.observed.map((obs, i) => {
    const row: Record<string, number | null> = {
      t: i + 1,
      observed: obs,
      q025: overlay.q025[i],
      q25: overlay.q25[i],
      median: overlay.median[i],
      q75: overlay.q75[i],
      q975: overlay.q975[i],
    };
    // Add spaghetti draws
    if (hasSpaghetti) {
      for (let d = 0; d < overlay.spaghetti_draws!.length; d++) {
        row[`draw_${d}`] = overlay.spaghetti_draws![d][i];
      }
    }
    return row;
  });

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="t"
            tick={{ fontSize: 11 }}
            label={{ value: "Time", position: "insideBottom", offset: -2, fontSize: 11 }}
          />
          <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => formatNumber(v, 1)} />
          <RechartsTooltip
            formatter={
              // biome-ignore lint/suspicious/noExplicitAny: recharts overload
              ((value: number, name: string) => {
                const labels: Record<string, string> = {
                  observed: "Observed",
                  median: "Median (y_rep)",
                  q025: "2.5%",
                  q975: "97.5%",
                  q25: "25%",
                  q75: "75%",
                };
                if (name.startsWith("draw_")) return [formatNumber(value, 2), "y_rep"];
                return [formatNumber(value, 2), labels[name] ?? name];
              }) as any
            }
            // biome-ignore lint/suspicious/noExplicitAny: recharts overload
            labelFormatter={((label: number) => `t = ${label}`) as any}
          />
          {/* 95% band (lightest) */}
          <Area
            dataKey="q975"
            stroke="none"
            fill="var(--primary)"
            fillOpacity={0.1}
            type="monotone"
          />
          <Area
            dataKey="q025"
            stroke="none"
            fill="var(--background)"
            fillOpacity={1}
            type="monotone"
          />
          {/* 50% band (darker) */}
          <Area
            dataKey="q75"
            stroke="none"
            fill="var(--primary)"
            fillOpacity={0.2}
            type="monotone"
          />
          <Area
            dataKey="q25"
            stroke="none"
            fill="var(--background)"
            fillOpacity={1}
            type="monotone"
          />
          {/* Spaghetti draws (thin, semi-transparent) */}
          {hasSpaghetti &&
            overlay.spaghetti_draws!.map((_, d) => (
              <Line
                key={`draw_${d}`}
                dataKey={`draw_${d}`}
                stroke="var(--primary)"
                strokeWidth={0.5}
                strokeOpacity={0.15}
                dot={false}
                type="monotone"
                name={`draw_${d}`}
                legendType="none"
              />
            ))}
          {/* Posterior predictive median */}
          <Line
            dataKey="median"
            stroke="var(--primary)"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            name="median"
          />
          {/* Observed data */}
          <Line
            dataKey="observed"
            stroke="var(--foreground)"
            strokeWidth={2}
            dot={{ r: 2, fill: "var(--foreground)" }}
            name="observed"
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
