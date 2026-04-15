"use client";

import { formatNumber } from "@/lib/utils/format";
import type { SensitivityAnalysisResult } from "@causal-ssm/api-types";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

const LOG_FLOOR = 1e-6;

function directionStatus(value: number): "pass" | "warn" | "fail" {
  if (value > 10) return "pass";
  if (value > 1) return "warn";
  return "fail";
}

function formatSV(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  if (Math.abs(value) < 0.01 || Math.abs(value) >= 1_000) {
    return value.toExponential(1);
  }
  return formatNumber(value, 2);
}

export function SensitivityDirectionsChart({ result }: { result: SensitivityAnalysisResult }) {
  const data = result.normalized_singular_values.map((normalizedSingularValue, index) => ({
    direction: index + 1,
    normalizedSingularValue,
    plotValue: Math.max(normalizedSingularValue, LOG_FLOOR),
    status: directionStatus(normalizedSingularValue),
  }));

  const passData = data.filter((entry) => entry.status === "pass");
  const warnData = data.filter((entry) => entry.status === "warn");
  const failData = data.filter((entry) => entry.status === "fail");

  return (
    <div className="space-y-2">
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 24, left: 10, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="direction"
              type="number"
              tick={{ fontSize: 10 }}
              label={{
                value: "Jacobian Direction",
                position: "insideBottom",
                offset: -2,
                fontSize: 10,
              }}
            />
            <YAxis
              dataKey="plotValue"
              type="number"
              scale="log"
              domain={[LOG_FLOOR, "auto"]}
              tick={{ fontSize: 10 }}
              tickFormatter={(value: number) => formatSV(value)}
              label={{
                value: "Normalized SV",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                fontSize: 10,
              }}
            />
            <RechartsTooltip
              content={({ payload }) => {
                if (!payload?.length) return null;
                const point = payload[0].payload as {
                  direction: number;
                  normalizedSingularValue: number;
                  status: "pass" | "warn" | "fail";
                };
                return (
                  <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                    <p>
                      Jacobian Direction: <span className="font-mono">{point.direction}</span>
                    </p>
                    <p>
                      Normalized SV:{" "}
                      <span className="font-mono">{formatSV(point.normalizedSingularValue)}</span>
                    </p>
                    <p className="capitalize text-muted-foreground">Status: {point.status}</p>
                  </div>
                );
              }}
            />
            <ReferenceLine
              y={1}
              stroke="var(--destructive)"
              strokeDasharray="4 4"
              label={{
                value: "1.0",
                position: "right",
                fontSize: 9,
                fill: "var(--destructive)",
              }}
            />
            <ReferenceLine
              y={10}
              stroke="var(--warning)"
              strokeDasharray="4 4"
              label={{
                value: "10.0",
                position: "right",
                fontSize: 9,
                fill: "var(--warning)",
              }}
            />
            <Scatter data={passData} fill="var(--primary)" fillOpacity={0.7} r={3} />
            <Scatter data={warnData} fill="var(--warning)" fillOpacity={0.85} r={3.5} />
            <Scatter data={failData} fill="var(--destructive)" fillOpacity={0.9} r={4} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
