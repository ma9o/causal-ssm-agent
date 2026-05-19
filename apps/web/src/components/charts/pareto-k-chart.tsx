"use client";

import { PARETO_K_FAIL, PARETO_K_WARN } from "@/lib/constants/diagnostics";
import { formatNumber } from "@/lib/utils/format";
import type { LOODiagnostics } from "@nof1-causal-lab/api-types";
import {
  CartesianGrid,
  Line,
  LineChart,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

interface ParetoKChartProps {
  loo: LOODiagnostics;
}

export function ParetoKChart({ loo }: ParetoKChartProps) {
  if (!loo.pareto_k || loo.pareto_k.length === 0) return null;

  const sorted = loo.pareto_k
    .map((k, i) => ({ k, timestep: i + 1 }))
    .sort((a, b) => b.k - a.k)
    .map((entry, rank) => ({ rank: rank + 1, k: entry.k, timestep: entry.timestep }));

  const nFail = sorted.filter((d) => d.k > PARETO_K_FAIL).length;
  const nWarn = sorted.filter((d) => d.k > PARETO_K_WARN && d.k <= PARETO_K_FAIL).length;

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-mono text-muted-foreground">Pareto k (sorted)</span>
        <span className="text-[10px] font-mono text-muted-foreground">
          {nFail} &gt; {PARETO_K_FAIL} · {nWarn} &gt; {PARETO_K_WARN} · n = {sorted.length}
        </span>
      </div>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={sorted} margin={{ top: 10, right: 40, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="rank"
              type="number"
              domain={[1, sorted.length]}
              tick={{ fontSize: 10 }}
              label={{ value: "Rank", position: "insideBottom", offset: -2, fontSize: 10 }}
            />
            <YAxis
              dataKey="k"
              tick={{ fontSize: 10 }}
              label={{
                value: "Pareto k",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                fontSize: 10,
              }}
            />
            <RechartsTooltip
              formatter={(value, _name, item) => {
                const ts = (item?.payload as { timestep?: number } | undefined)?.timestep;
                return [
                  `${formatNumber(Number(value), 3)}${ts != null ? ` (timestep ${ts})` : ""}`,
                  "Pareto k",
                ];
              }}
              labelFormatter={(label) => `rank ${label}`}
            />
            <ReferenceLine
              y={PARETO_K_FAIL}
              stroke="var(--destructive)"
              strokeDasharray="4 4"
              label={{
                value: `k = ${PARETO_K_FAIL}`,
                position: "right",
                fontSize: 9,
                fill: "var(--destructive)",
              }}
            />
            <ReferenceLine
              y={PARETO_K_WARN}
              stroke="var(--warning)"
              strokeDasharray="4 4"
              label={{
                value: `k = ${PARETO_K_WARN}`,
                position: "right",
                fontSize: 9,
                fill: "var(--warning)",
              }}
            />
            <Line
              dataKey="k"
              type="linear"
              stroke="var(--primary)"
              strokeWidth={1.25}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">
        Pareto k diagnostic sorted from largest to smallest. Timesteps with k &gt; {PARETO_K_FAIL}{" "}
        (left edge) are highly influential and the LOO estimate may be unreliable. Hover any rank to
        recover the original timestep.
      </p>
    </div>
  );
}
