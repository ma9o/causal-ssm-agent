"use client";

import { formatNumber } from "@/lib/utils/format";
import { Badge } from "@/components/ui/badge";
import type { EnergyDiagnostics } from "@causal-ssm/api-types";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

interface EnergyChartProps {
  energy: EnergyDiagnostics;
}

export function EnergyChart({ energy }: EnergyChartProps) {
  // Merge energy and transition histograms into a single dataset
  const data = energy.energy_hist.bin_centers.map((x, i) => ({
    x,
    energy: energy.energy_hist.density[i],
  }));

  const transData = energy.energy_transition_hist.bin_centers.map((x, i) => ({
    x,
    transition: energy.energy_transition_hist.density[i],
  }));

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
        {/* Marginal energy distribution */}
        <div>
          <span className="text-xs font-mono text-muted-foreground">Marginal Energy E</span>
          <div className="h-36 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="x" tick={{ fontSize: 9 }} tickFormatter={(v: number) => formatNumber(v, 0)} />
                <YAxis tick={{ fontSize: 9 }} tickFormatter={(v: number) => formatNumber(v, 2)} />
                <RechartsTooltip
                  formatter={
                    // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                    ((value: number) => [formatNumber(value, 4), "Density"]) as any
                  }
                />
                <Area
                  dataKey="energy"
                  stroke="var(--primary)"
                  fill="var(--primary)"
                  fillOpacity={0.2}
                  type="monotone"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
        {/* Energy transition distribution */}
        <div>
          <span className="text-xs font-mono text-muted-foreground">Energy Transition dE</span>
          <div className="h-36 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={transData} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="x" tick={{ fontSize: 9 }} tickFormatter={(v: number) => formatNumber(v, 0)} />
                <YAxis tick={{ fontSize: 9 }} tickFormatter={(v: number) => formatNumber(v, 2)} />
                <RechartsTooltip
                  formatter={
                    // biome-ignore lint/suspicious/noExplicitAny: recharts overload
                    ((value: number) => [formatNumber(value, 4), "Density"]) as any
                  }
                />
                <Area
                  dataKey="transition"
                  stroke="var(--chart-2)"
                  fill="var(--chart-2)"
                  fillOpacity={0.2}
                  type="monotone"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
