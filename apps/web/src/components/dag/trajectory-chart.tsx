"use client";

import {
  CartesianGrid,
  ComposedChart,
  Line,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { formatNumber } from "@/lib/utils/format";
import {
  getEffectTrajectoryDays,
  getNodeActionSeries,
  getNodeReferenceSeries,
} from "./intervention-dag-semantics";
import type { Stage6SimulationResult } from "./intervention-dag-types";
import type { DagMetric } from "./intervention-dag-view-model";

interface SeriesSpec {
  primary: number[];
  primaryLabel: string;
  baseline?: number[];
  baselineLabel?: string;
}

function seriesForMetric(result: Stage6SimulationResult, metric: DagMetric): SeriesSpec {
  const outcome = result.outcome;
  if (metric === "action") {
    return {
      primary: getNodeActionSeries(result, outcome) ?? [],
      primaryLabel: "Action path",
      baseline: getNodeReferenceSeries(result, outcome) ?? undefined,
      baselineLabel: "Reference",
    };
  }
  if (metric === "reference") {
    return {
      primary: getNodeReferenceSeries(result, outcome) ?? [],
      primaryLabel: "Reference path",
    };
  }
  return {
    primary: (result.effect_trajectory ?? []).map((point) => point.effect),
    primaryLabel: "Effect Δ",
  };
}

/**
 * Temporal view of the focused outcome under a fixed simulation. The metric
 * toggle re-slices which trajectory is plotted (effect Δ / action path / reference).
 */
export function TrajectoryChart({
  result,
  metric,
  height = 180,
}: {
  result: Stage6SimulationResult;
  metric: DagMetric;
  height?: number;
}) {
  const days = getEffectTrajectoryDays(result);
  const { primary, primaryLabel, baseline, baselineLabel } = seriesForMetric(result, metric);
  if (days.length === 0 || primary.length === 0) {
    return null;
  }

  const data = days.map((day, index) => ({
    day,
    primary: primary[index] ?? null,
    baseline: baseline?.[index] ?? null,
  }));

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <span
            className="inline-block h-0.5 w-3 rounded"
            style={{ backgroundColor: "var(--primary)" }}
          />
          {primaryLabel} on {result.outcome}
        </span>
        {baseline ? (
          <span className="inline-flex items-center gap-1">
            <span
              className="inline-block h-0.5 w-3 rounded"
              style={{ backgroundColor: "var(--chart-2)" }}
            />
            {baselineLabel}
          </span>
        ) : null}
      </div>
      <div className="w-full" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 15, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="day"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fontSize: 9 }}
              tickFormatter={(v: number) => `${formatNumber(v, 0)}d`}
            />
            <YAxis tick={{ fontSize: 9 }} tickFormatter={(v: number) => formatNumber(v, 2)} />
            <RechartsTooltip
              formatter={(value: number | string | undefined, name: string | undefined) => {
                const numeric = typeof value === "number" ? value : Number(value);
                return [Number.isFinite(numeric) ? formatNumber(numeric, 3) : "--", name ?? ""] as [
                  string,
                  string,
                ];
              }}
              labelFormatter={(label: unknown) => {
                const numeric = typeof label === "number" ? label : Number(label);
                return Number.isFinite(numeric) ? `day ${formatNumber(numeric, 0)}` : "day --";
              }}
              contentStyle={{ fontSize: 11, padding: "2px 8px" }}
            />
            {metric === "effect" ? (
              <ReferenceLine y={0} strokeDasharray="4 3" className="stroke-muted-foreground/60" />
            ) : null}
            {baseline ? (
              <Line
                name="reference"
                dataKey="baseline"
                type="monotone"
                stroke="var(--chart-2)"
                strokeWidth={1}
                strokeOpacity={0.5}
                dot={false}
                isAnimationActive={false}
              />
            ) : null}
            <Line
              name="effect"
              dataKey="primary"
              type="monotone"
              stroke="var(--primary)"
              strokeWidth={1.75}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
