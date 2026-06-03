"use client";

import {
  Bar,
  ComposedChart,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { cn } from "@/lib/utils";
import { formatNumber } from "@/lib/utils/format";
import { buildHistogram } from "@/lib/utils/histogram";

type ManifestEffects = Record<string, number | undefined> | null | undefined;

/** Posterior-draw histogram with zero and mean reference lines. */
export function PosteriorHistogram({
  draws,
  mean,
  className,
}: {
  draws: number[];
  mean: number | null;
  className?: string;
}) {
  if (draws.length === 0) return <span className="text-xs text-muted-foreground">--</span>;

  const bins = buildHistogram(draws, Math.min(25, Math.ceil(Math.sqrt(draws.length))));

  return (
    <div className={cn("ml-auto h-16 w-44", className)}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={bins} margin={{ top: 2, right: 4, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="binCenter"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v: number) => formatNumber(v, 2)}
            tick={{ fontSize: 9 }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis hide />
          <RechartsTooltip
            formatter={(v: number | string | undefined) => {
              const numeric = typeof v === "number" ? v : Number(v);
              return [Number.isFinite(numeric) ? formatNumber(numeric, 0) : "--", "count"] as const;
            }}
            labelFormatter={(l: unknown) => {
              const numeric = typeof l === "number" ? l : Number(l);
              return Number.isFinite(numeric) ? `τ = ${formatNumber(numeric, 3)}` : "τ = --";
            }}
            contentStyle={{ fontSize: 10, padding: "2px 6px" }}
          />
          <ReferenceLine x={0} strokeDasharray="4 3" className="stroke-muted-foreground/60" />
          {mean !== null && <ReferenceLine x={mean} stroke="var(--foreground)" strokeWidth={1.5} />}
          <Bar dataKey="count" fill="var(--muted-foreground)" opacity={0.35} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Indicator-level projection of an outcome effect through the measurement loadings. */
export function ManifestProjection({
  manifestEffects,
  className,
}: {
  manifestEffects: ManifestEffects;
  className?: string;
}) {
  const entries = Object.entries(manifestEffects ?? {})
    .filter((entry): entry is [string, number] => typeof entry[1] === "number")
    .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]));
  if (entries.length === 0) return null;

  return (
    <div className={cn("flex items-start gap-4", className)}>
      <span className="shrink-0 pt-px text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        Indicators
      </span>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
        {entries.map(([indicator, value]) => (
          <span key={indicator} className="inline-flex items-baseline gap-1.5">
            <span className="text-muted-foreground">{indicator}</span>
            <span className="font-mono text-xs tabular-nums">
              {value >= 0 ? "+" : ""}
              {formatNumber(value)}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
