"use client";

import { diagnosisColor, diagnosisLabel } from "@/lib/constants/charts";
import { formatNumber } from "@/lib/utils/format";
import type { PowerScalingResult } from "@causal-ssm/api-types";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

interface PowerScalingScatterProps {
  results: PowerScalingResult[];
}



export function PowerScalingScatter({ results }: PowerScalingScatterProps) {
  // Group by diagnosis for coloring
  const groups = new Map<string, PowerScalingResult[]>();
  for (const r of results) {
    const existing = groups.get(r.diagnosis) ?? [];
    existing.push(r);
    groups.set(r.diagnosis, existing);
  }

  return (
    <div className="min-h-56 h-full w-full flex flex-col">
      <ResponsiveContainer width="100%" height="100%" className="flex-1">
        <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="prior_sensitivity"
            type="number"
            domain={[0, 1]}
            tick={{ fontSize: 11 }}
            name="Prior Sensitivity"
            label={{
              value: "Prior Sensitivity",
              position: "insideBottom",
              offset: -15,
              fontSize: 11,
            }}
          />
          <YAxis
            dataKey="likelihood_sensitivity"
            type="number"
            domain={[0, 1]}
            tick={{ fontSize: 11 }}
            name="Likelihood Sensitivity"
            label={{
              value: "Lik. Sensitivity",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              fontSize: 11,
            }}
          />
          <ZAxis range={[80, 80]} />
          <RechartsTooltip
            content={({ payload }) => {
              if (!payload?.length) return null;
              const d = payload[0].payload as PowerScalingResult;
              return (
                <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                  <p className="font-mono font-medium">{d.parameter}</p>
                  <p>Prior sens: {formatNumber(d.prior_sensitivity, 2)}</p>
                  <p>Lik. sens: {formatNumber(d.likelihood_sensitivity, 2)}</p>
                  <p>{diagnosisLabel[d.diagnosis] ?? d.diagnosis}</p>
                </div>
              );
            }}
          />
          {/* Threshold guides */}
          <ReferenceLine
            x={0.05}
            stroke="var(--muted-foreground)"
            strokeDasharray="4 4"
            strokeOpacity={0.5}
          />
          <ReferenceLine
            y={0.05}
            stroke="var(--muted-foreground)"
            strokeDasharray="4 4"
            strokeOpacity={0.5}
          />
          {Array.from(groups.entries()).map(([diagnosis, items]) => (
            <Scatter
              key={diagnosis}
              name={diagnosisLabel[diagnosis] ?? diagnosis}
              data={items}
              fill={diagnosisColor[diagnosis] ?? "var(--primary)"}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex items-center justify-center gap-4 pt-1 text-[10px] text-muted-foreground">
        {(["well_identified", "prior_dominated", "prior_data_conflict"] as const).map((d) => (
          <span key={d} className="inline-flex items-center gap-1">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: diagnosisColor[d] }}
            />
            {diagnosisLabel[d]}
          </span>
        ))}
      </div>
    </div>
  );
}
