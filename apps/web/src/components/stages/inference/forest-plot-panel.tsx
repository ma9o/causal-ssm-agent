"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber } from "@/lib/utils/format";
import type { TreatmentEffect } from "@causal-ssm/api-types";
import { useMemo } from "react";
import {
  CartesianGrid,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ticks as d3Ticks, min as d3Min, max as d3Max } from "d3-array";

interface ForestPlotPanelProps {
  results: TreatmentEffect[];
}

const ROW_HEIGHT = 32;
const PADDING_TOP = 4;
const DIAMOND_H = 5;
const DIAMOND_W = 6;

interface ChartRow {
  x: number;
  y: number;
  errorX: [number, number];
  treatment: string;
  effect_size: number;
  ci_lower: number;
  ci_upper: number;
  prob_positive?: number;
}

function DiamondDot({ cx, cy }: { cx?: number; cy?: number }) {
  if (cx == null || cy == null) return null;
  return (
    <polygon
      points={`${cx},${cy - DIAMOND_H} ${cx + DIAMOND_W},${cy} ${cx},${cy + DIAMOND_H} ${cx - DIAMOND_W},${cy}`}
      fill="var(--primary)"
    />
  );
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ChartRow }> }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-md border bg-popover px-3 py-2 text-sm shadow-md">
      <div className="mb-1 font-medium">{d.treatment}</div>
      <div className="space-y-0.5 font-mono text-xs text-muted-foreground">
        <div>{"\u03C4\u0302"} = {formatNumber(d.effect_size, 3)}</div>
        <div>95% CI [{formatNumber(d.ci_lower, 3)}, {formatNumber(d.ci_upper, 3)}]</div>
        {d.prob_positive !== undefined && (
          <div>P(τ&gt;0) = {formatNumber(d.prob_positive, 3)}</div>
        )}
      </div>
    </div>
  );
}

export function ForestPlotPanel({ results }: ForestPlotPanelProps) {
  if (results.length === 0) return null;
  const plottable = results.filter(
    (r) => r.effect_size !== null && r.credible_interval !== null,
  );
  if (plottable.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Forest Plot</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          No numeric treatment effects available to plot.
        </CardContent>
      </Card>
    );
  }

  const sorted = useMemo(
    () => [...plottable].sort((a, b) => (b.effect_size as number) - (a.effect_size as number)),
    [plottable],
  );

  const { domainMin, domainMax } = useMemo(() => {
    const allVals = sorted.flatMap((r) => {
      const ci = r.credible_interval as [number, number];
      return [ci[0], ci[1], 0];
    });
    const lo = d3Min(allVals) ?? 0;
    const hi = d3Max(allVals) ?? 0;
    const pad = (hi - lo) * 0.15;
    return { domainMin: lo - pad, domainMax: hi + pad };
  }, [sorted]);

  const ticks = useMemo(
    () => d3Ticks(domainMin, domainMax, 5),
    [domainMin, domainMax],
  );

  const rows: ChartRow[] = useMemo(
    () =>
      sorted.map((r, i) => {
        const effect = r.effect_size as number;
        const ci = r.credible_interval as [number, number];
        return {
          x: effect,
          y: i,
          errorX: [effect - ci[0], ci[1] - effect],
          treatment: r.treatment,
          effect_size: effect,
          ci_lower: ci[0],
          ci_upper: ci[1],
          prob_positive: r.prob_positive ?? undefined,
        };
      }),
    [sorted],
  );

  const svgHeight = PADDING_TOP + sorted.length * ROW_HEIGHT + 20;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Forest Plot</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex w-full gap-0">
          {/* Treatment labels column */}
          <div className="shrink-0 pr-3" style={{ paddingTop: PADDING_TOP }}>
            {sorted.map((r) => (
              <div
                key={r.treatment}
                className="flex items-center text-sm text-muted-foreground"
                style={{ height: ROW_HEIGHT }}
              >
                <span className="max-w-[140px] truncate">{r.treatment}</span>
              </div>
            ))}
          </div>

          {/* Chart area */}
          <div className="min-w-0 flex-1">
            <ResponsiveContainer width="100%" height={svgHeight}>
              <ScatterChart margin={{ top: PADDING_TOP, right: 0, bottom: 0, left: 0 }}>
                <CartesianGrid
                  vertical
                  horizontal={false}
                  strokeDasharray="none"
                  className="stroke-muted"
                />
                <XAxis
                  type="number"
                  dataKey="x"
                  domain={[domainMin, domainMax]}
                  ticks={ticks}
                  tickFormatter={(t: number) => formatNumber(t, 2)}
                  fontSize={11}
                  className="fill-muted-foreground"
                  axisLine={false}
                  tickLine={{ className: "stroke-muted-foreground" }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  hide
                  domain={[-0.5, sorted.length - 0.5]}
                  reversed
                />
                <ReferenceLine
                  x={0}
                  strokeDasharray="4 3"
                  className="stroke-muted-foreground/60"
                />
                <RechartTooltip
                  content={<CustomTooltip />}
                  cursor={false}
                />
                <Scatter
                  data={rows}
                  shape={<DiamondDot />}
                  isAnimationActive={false}
                >
                  <ErrorBar
                    dataKey="errorX"
                    direction="x"
                    width={10}
                    stroke="var(--primary)"
                    strokeWidth={1.5}
                  />
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Numeric summary column */}
          <div className="shrink-0 pl-3" style={{ paddingTop: PADDING_TOP }}>
            {sorted.map((r) => {
              const ci = r.credible_interval as [number, number];
              const effect = r.effect_size as number;
              return (
                <div
                  key={r.treatment}
                  className="flex items-center font-mono text-xs text-foreground"
                  style={{ height: ROW_HEIGHT }}
                >
                  <span className="tabular-nums whitespace-nowrap">
                    {formatNumber(effect, 2)}{" "}
                    <span className="text-muted-foreground">
                      [{formatNumber(ci[0], 2)},{" "}
                      {formatNumber(ci[1], 2)}]
                    </span>
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
