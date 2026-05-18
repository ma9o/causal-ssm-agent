"use client";

import { SensitivityDirectionsChart } from "@/components/stages/parametric-id/sensitivity-directions-chart";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { formatNumber } from "@/lib/utils/format";
import type {
  SensitivityAnalysisResult,
  SensitivityDirection,
  SensitivityDirectionLoading,
  SensitivityEntry,
} from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

const directionCol = createColumnHelper<SensitivityDirection>();
const parameterCol = createColumnHelper<SensitivityEntry>();

function formatSV(v: number | null | undefined) {
  if (v == null) return "—";
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  return Math.abs(v) < 0.01 || Math.abs(v) >= 1_000 ? v.toExponential(1) : formatNumber(v, 4);
}

function statusToSeverity(status: string): "fail" | "warn" | undefined {
  if (status === "fail") return "fail";
  if (status === "warn") return "warn";
  return undefined;
}

function LoadingsBarChart({ loadings }: { loadings: SensitivityDirectionLoading[] }) {
  if (loadings.length === 0) return <span className="text-xs text-muted-foreground">—</span>;

  const maxAbs = Math.max(...loadings.map((l) => Math.abs(l.loading)));
  const data = loadings.map((l) => ({
    name: l.interpretable_parameter,
    loading: l.loading,
    opacity: maxAbs > 0 ? 0.15 + 0.7 * (Math.abs(l.loading) / maxAbs) : 0.5,
  }));

  return (
    <div className="h-14 w-40">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 2, right: 4, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="name"
            tick={false}
            axisLine={{ stroke: "var(--border)" }}
            height={2}
          />
          <YAxis
            tick={{ fontSize: 8 }}
            tickLine={false}
            axisLine={false}
            width={32}
            tickFormatter={(v: number) => formatNumber(v, 2)}
          />
          <RechartsTooltip
            formatter={(v: number | string | undefined) => {
              const n = typeof v === "number" ? v : Number(v);
              return [Number.isFinite(n) ? formatSV(n) : "—", "loading"] as const;
            }}
            labelFormatter={(label: unknown) => String(label)}
            contentStyle={{ fontSize: 10, padding: "2px 6px" }}
          />
          <ReferenceLine y={0} stroke="var(--border)" />
          <Bar dataKey="loading" isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell key={i} fill="var(--primary)" fillOpacity={d.opacity} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const weakDirectionColumns = [
  directionCol.accessor("index", {
    header: "Jacobian Direction",
    cell: (info) => <span className="font-medium">d{info.getValue()}</span>,
    meta: { mono: true },
  }),
  directionCol.accessor("normalized_singular_value", {
    header: () => (
      <HeaderWithTooltip
        label="Normalized SV"
        tooltip="Jacobian-direction singular value of the normalized sensitivity matrix. Values <= 1 indicate deficient local Jacobian directions; values between 1 and 10 are borderline."
      />
    ),
    cell: (info) => formatSV(info.getValue()),
    meta: {
      mono: true,
      severity: (_v: number, row: SensitivityDirection) => statusToSeverity(row.status),
    },
  }),
  directionCol.accessor("singular_value", {
    header: () => (
      <HeaderWithTooltip
        label="Raw SV"
        tooltip="Singular value before prior-SD and observation-noise normalization. Useful for scale context, but not the thresholding quantity."
      />
    ),
    cell: (info) => formatSV(info.getValue()),
    meta: { mono: true },
  }),
  directionCol.accessor("top_loadings", {
    header: () => (
      <HeaderWithTooltip
        label="Normalized Sensitivity Eigenvector Components"
        tooltip="Largest signed components of the eigenvector of the normalized sensitivity Gram matrix for this weak local parameter combination. These are the named parameters most involved in the poorly constrained direction."
      />
    ),
    cell: (info) => <LoadingsBarChart loadings={info.getValue()} />,
  }),
];

const fullParameterColumns = [
  parameterCol.accessor("parameter", {
    header: "Parameter",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
    meta: { mono: true },
  }),
  parameterCol.accessor("interpretable_parameter", {
    header: () => (
      <HeaderWithTooltip
        label="Interpretable Parameter"
        tooltip="Semantic parameter name resolved from the Stage 4 model specification when available. This maps compiled scalar sites back to user-facing parameter names."
      />
    ),
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
    meta: { mono: true },
  }),
  parameterCol.accessor("sensitivity_norm", {
    header: () => (
      <HeaderWithTooltip
        label="Sensitivity"
        tooltip="Output sensitivity norm: the L2 norm of the Jacobian column for this parameter. Higher values indicate larger changes in the emitted-observation moment summary when this parameter is perturbed."
      />
    ),
    cell: (info) => formatNumber(info.getValue(), 4),
    meta: { mono: true },
  }),
  parameterCol.accessor("effective_sv", {
    header: () => (
      <HeaderWithTooltip
        label="Effective SV"
        tooltip="Minimum singular value among raw Jacobian directions where this parameter has substantial weight. Captures aliasing in the unnormalized sensitivity geometry."
      />
    ),
    cell: (info) => formatSV(info.getValue()),
    meta: {
      mono: true,
      severity: (_v: number, row: SensitivityEntry) => statusToSeverity(row.sv_status),
    },
  }),
  parameterCol.accessor("normalized_effective_sv", {
    header: () => (
      <HeaderWithTooltip
        label="Normalized SV"
        tooltip="Minimum singular value among normalized Jacobian directions where this parameter has substantial weight. Thresholds: >10 pass, >1 warn, <=1 fail."
      />
    ),
    cell: (info) => formatSV(info.getValue()),
    meta: {
      mono: true,
      severity: (_v: number, row: SensitivityEntry) => statusToSeverity(row.normalized_sv_status),
    },
  }),
];

export function SensitivityAnalysisTable({ result }: { result: SensitivityAnalysisResult }) {
  const [openSections, setOpenSections] = useState<string[]>([]);
  const directionCount = result.normalized_singular_values.length;
  const weakDirectionCount = result.deficiency_count;
  const borderlineDirectionCount = result.weak_directions.filter(
    (row) => row.status === "warn",
  ).length;
  const strongDirectionCount = directionCount - weakDirectionCount - borderlineDirectionCount;

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="flex flex-wrap items-center justify-center gap-x-3 gap-y-2 py-1 text-sm">
          <span className="inline-flex items-center gap-1.5 font-medium">
            Direction-Level Sensitivity
            <StatTooltip explanation="Stage 4b first checks local identifiability through the SVD of the normalized sensitivity matrix. Each direction is an orthogonal parameter combination, not a single parameter." />
          </span>

          <Badge variant={weakDirectionCount > 0 ? "destructive" : "success"}>
            weak directions: {weakDirectionCount}/{directionCount}
          </Badge>
          {borderlineDirectionCount > 0 && (
            <Badge variant="warning">borderline: {borderlineDirectionCount}</Badge>
          )}
          {strongDirectionCount > 0 && (
            <Badge variant="secondary">well-conditioned: {strongDirectionCount}</Badge>
          )}
          <Badge variant="secondary">params: {result.n_parameters}</Badge>
          <Badge variant="secondary">draws: {result.n_draws}</Badge>
        </CardContent>
      </Card>

      <div className="space-y-4">
        <section className="space-y-2">
          <div>
            <h4 className="text-sm font-semibold">Direction Spectrum</h4>
          </div>
          <SensitivityDirectionsChart result={result} />
        </section>

        <section className="space-y-2">
          <div>
            <h4 className="text-sm font-semibold">Weakest Directions</h4>
            <p className="text-sm text-muted-foreground">
              Weak local parameter combinations, ordered by normalized singular value.
            </p>
          </div>
          {result.weak_directions.length > 0 ? (
            <InfoTable
              columns={weakDirectionColumns as ColumnDef<SensitivityDirection, unknown>[]}
              data={result.weak_directions}
              filtering={false}
              maxHeight="max-h-[22rem]"
              estimateRowHeight={68}
            />
          ) : (
            <div className="rounded-md border border-dashed px-4 py-6 text-sm text-muted-foreground">
              No weak or borderline local directions were detected under the current thresholds.
            </div>
          )}
        </section>
      </div>

      <Accordion multiple value={openSections} onValueChange={setOpenSections}>
        <AccordionItem value="all-parameters">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              All Parameter Diagnostics
              <Badge variant="outline">{result.per_parameter.length} rows</Badge>
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <InfoTable
              columns={fullParameterColumns as ColumnDef<SensitivityEntry, unknown>[]}
              data={result.per_parameter}
              maxHeight="max-h-[32rem]"
            />
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
