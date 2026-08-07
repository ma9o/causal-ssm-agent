"use client";

import { Badge } from "@/components/ui/badge";
import { formatNumber } from "@/lib/utils/format";
import type { EnergyDiagnostics } from "@nof1-causal-lab/api-types";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

interface EnergyChartProps {
  energy: EnergyDiagnostics;
}

function DensityHistogram({
  title,
  histogram,
  color,
}: {
  title: string;
  histogram: EnergyDiagnostics["energy_hist"];
  color: string;
}) {
  const data = histogram.bin_centers.map((x, index) => ({
    x,
    density: histogram.density[index],
  }));

  return (
    <div>
      <span className="text-xs font-mono text-muted-foreground">{title}</span>
      <div className="h-36 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="x"
              tick={{ fontSize: 9 }}
              tickFormatter={(value: number) => formatNumber(value, 0)}
            />
            <YAxis
              tick={{ fontSize: 9 }}
              tickFormatter={(value: number) => formatNumber(value, 2)}
            />
            <RechartsTooltip formatter={(value) => [formatNumber(Number(value), 4), "Density"]} />
            <Area dataKey="density" stroke={color} fill={color} fillOpacity={0.2} type="monotone" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function EnergyChart({ energy }: EnergyChartProps) {
  const minBfmi = Math.min(...energy.bfmi);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">BFMI:</span>
        {energy.bfmi.map((b, i) => (
          <Badge
            key={`bfmi-${
              // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
              i
            }`}
            variant={b < 0.3 ? "destructive" : "success"}
          >
            Chain {i + 1}: {formatNumber(b, 2)}
          </Badge>
        ))}
        {minBfmi < 0.3 && (
          <span className="text-xs text-destructive">Low BFMI indicates poor exploration</span>
        )}
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <DensityHistogram
          title="Marginal Energy E"
          histogram={energy.energy_hist}
          color="var(--primary)"
        />
        <DensityHistogram
          title="Energy Transition dE"
          histogram={energy.energy_transition_hist}
          color="var(--chart-2)"
        />
      </div>
    </div>
  );
}
