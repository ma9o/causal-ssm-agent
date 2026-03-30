"use client";

import { memo } from "react";

interface SparklineProps {
  /** Main data series */
  series: number[];
  /** How many points to draw (animation progress) */
  visibleCount: number;
  /** X-axis day values for proportional spacing */
  days: number[];
  /** Line color (CSS value) */
  color: string;
  /** Optional baseline series shown full-length and dimmed (Rung 3 factual) */
  baselineSeries?: number[];
  /** Draw a dashed zero reference line (Rung 2 mode) */
  showZero?: boolean;
  width?: number;
  height?: number;
}

function toPoints(
  series: number[],
  count: number,
  days: number[],
  xScale: (d: number) => number,
  yScale: (v: number) => number,
): string {
  const pts: string[] = [];
  const n = Math.min(count, series.length, days.length);
  for (let i = 0; i < n; i++) {
    pts.push(`${xScale(days[i]).toFixed(1)},${yScale(series[i]).toFixed(1)}`);
  }
  return pts.join(" ");
}

function SparklineInner({
  series,
  visibleCount,
  days,
  color,
  baselineSeries,
  showZero = false,
  width = 180,
  height = 28,
}: SparklineProps) {
  if (series.length === 0 || days.length === 0 || visibleCount <= 0) return null;

  const px = 2;
  const py = 3;
  const iw = width - px * 2;
  const ih = height - py * 2;

  // Y domain — include all data so both series share the same scale
  const allVals = [...series];
  if (baselineSeries) allVals.push(...baselineSeries);
  if (showZero) allVals.push(0);

  let yMin = Math.min(...allVals);
  let yMax = Math.max(...allVals);
  if (yMax - yMin < 0.02) {
    const mid = (yMax + yMin) / 2;
    yMin = mid - 0.1;
    yMax = mid + 0.1;
  }

  const dayMin = days[0];
  const dayMax = days[days.length - 1];
  const dr = dayMax - dayMin || 1;

  const xScale = (d: number) => px + ((d - dayMin) / dr) * iw;
  const yScale = (v: number) => py + (1 - (v - yMin) / (yMax - yMin)) * ih;

  const dotIdx = Math.min(visibleCount - 1, series.length - 1);

  return (
    <svg
      width={width}
      height={height}
      className="overflow-visible"
      aria-hidden="true"
    >
      {/* Zero reference line */}
      {showZero && yMin <= 0 && yMax >= 0 && (
        <line
          x1={px}
          y1={yScale(0)}
          x2={width - px}
          y2={yScale(0)}
          stroke="currentColor"
          strokeWidth={0.5}
          strokeDasharray="2,3"
          opacity={0.2}
        />
      )}

      {/* Baseline series (full length, dimmed) */}
      {baselineSeries && baselineSeries.length > 1 && (
        <polyline
          points={toPoints(
            baselineSeries,
            baselineSeries.length,
            days,
            xScale,
            yScale,
          )}
          fill="none"
          stroke="currentColor"
          strokeWidth={1}
          opacity={0.2}
        />
      )}

      {/* Main series up to visibleCount */}
      {visibleCount > 1 && (
        <polyline
          points={toPoints(series, visibleCount, days, xScale, yScale)}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      )}

      {/* Current-point dot */}
      {dotIdx >= 0 && (
        <circle
          cx={xScale(days[dotIdx])}
          cy={yScale(series[dotIdx])}
          r={2}
          fill={color}
        />
      )}
    </svg>
  );
}

export const Sparkline = memo(SparklineInner);
