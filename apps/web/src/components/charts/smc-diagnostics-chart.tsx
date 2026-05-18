"use client";

import { Badge } from "@/components/ui/badge";
import { formatNumber } from "@/lib/utils/format";
import type { SMCDiagnostics } from "@nof1-causal-lab/api-types";
import {
  CartesianGrid,
  Line,
  LineChart,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

interface SMCDiagnosticsChartProps {
  diagnostics: SMCDiagnostics;
}

export function SMCDiagnosticsChart({ diagnostics }: SMCDiagnosticsChartProps) {
  const { beta_schedule, ess_history, accept_rates, n_levels, n_particles } = diagnostics;
  const converged = beta_schedule.length > 0 && beta_schedule[beta_schedule.length - 1] >= 1.0;

  const data = beta_schedule.map((beta, i) => ({
    level: i + 1,
    beta,
    ess: ess_history[i] ?? 0,
    accept: (accept_rates[i] ?? 0) * 100,
  }));

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        <Badge variant="outline">{n_particles} particles</Badge>
        <Badge variant="outline">{n_levels} levels</Badge>
        <Badge variant={converged ? "success" : "destructive"}>
          {converged ? "Converged" : "Did not converge"}
        </Badge>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Beta + ESS chart */}
        <div>
          <h4 className="mb-1 text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Tempering schedule &amp; ESS
          </h4>
          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="level"
                  tick={{ fontSize: 11 }}
                  label={{ value: "Level", position: "insideBottom", offset: -2, fontSize: 11 }}
                />
                <YAxis
                  yAxisId="beta"
                  tick={{ fontSize: 11 }}
                  domain={[0, 1]}
                  label={{
                    value: "β",
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                    fontSize: 11,
                  }}
                />
                <YAxis
                  yAxisId="ess"
                  orientation="right"
                  tick={{ fontSize: 11 }}
                  label={{
                    value: "ESS",
                    angle: 90,
                    position: "insideRight",
                    offset: 10,
                    fontSize: 11,
                  }}
                />
                <RechartsTooltip
                  formatter={(value, name) => [
                    formatNumber(Number(value), 2),
                    name === "beta" ? "β" : "ESS",
                  ]}
                  labelFormatter={(label) => `Level ${label}`}
                />
                <Line
                  yAxisId="beta"
                  dataKey="beta"
                  stroke="var(--primary)"
                  strokeWidth={1.5}
                  dot={false}
                  type="monotone"
                />
                <Line
                  yAxisId="ess"
                  dataKey="ess"
                  stroke="var(--chart-2, hsl(var(--muted-foreground)))"
                  strokeWidth={1.5}
                  dot={false}
                  type="monotone"
                  strokeDasharray="4 2"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Acceptance rate chart */}
        <div>
          <h4 className="mb-1 text-xs font-medium text-muted-foreground uppercase tracking-wide">
            MH acceptance rate
          </h4>
          <div className="h-44 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="level"
                  tick={{ fontSize: 11 }}
                  label={{ value: "Level", position: "insideBottom", offset: -2, fontSize: 11 }}
                />
                <YAxis
                  tick={{ fontSize: 11 }}
                  domain={[0, 100]}
                  tickFormatter={(v: number) => `${v}%`}
                  label={{
                    value: "Accept %",
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                    fontSize: 11,
                  }}
                />
                <RechartsTooltip
                  formatter={(value) => [`${formatNumber(Number(value), 1)}%`, "Accept"]}
                  labelFormatter={(label) => `Level ${label}`}
                />
                <Line
                  dataKey="accept"
                  stroke="var(--primary)"
                  strokeWidth={1.5}
                  dot={false}
                  type="monotone"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
