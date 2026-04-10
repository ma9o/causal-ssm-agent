"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { formatNumber } from "@/lib/utils/format";
import type { SensitivityAnalysisResult, SensitivityEntry } from "@causal-ssm/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";

const col = createColumnHelper<SensitivityEntry>();

function formatSV(v: number | null | undefined) {
  if (v == null) return "—";
  return v < 0.01 ? v.toExponential(1) : formatNumber(v, 4);
}

function statusToSeverity(status: string): "fail" | "warn" | undefined {
  if (status === "fail") return "fail";
  if (status === "warn") return "warn";
  return undefined;
}

const columns = [
  col.accessor("parameter", {
    header: "Parameter",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
    meta: { mono: true },
  }),
  col.accessor("interpretable_parameter", {
    header: () => (
      <HeaderWithTooltip
        label="Interpretable Parameter"
        tooltip="Semantic parameter name resolved from the Stage 4 model specification when available. This maps compiled scalar sites like drift_offdiag_pop[0] back to user-facing parameters like beta_sleep_stress."
      />
    ),
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
    meta: { mono: true },
  }),
  col.accessor("sensitivity_norm", {
    header: () => (
      <HeaderWithTooltip
        label="Sensitivity"
        tooltip="Output sensitivity norm: the L2 norm of the Jacobian column for this parameter. Measures how much the emitted-observation moment summary changes when this parameter is perturbed. Higher values indicate stronger identifiability from data."
      />
    ),
    cell: (info) => formatNumber(info.getValue(), 4),
    meta: { mono: true },
  }),
  col.accessor("effective_sv", {
    header: () => (
      <HeaderWithTooltip
        label="Effective SV"
        tooltip="Effective singular value: the minimum singular value among SVD directions where this parameter has significant weight. Captures aliasing — two parameters can each have high sensitivity norms but share a near-singular direction, making their individual effects indistinguishable from data. Thresholds: >10⁻³·max = pass, >10⁻⁶·max = warn, ≤10⁻⁶·max = fail (3-decade gap convention)."
      />
    ),
    cell: (info) => formatSV(info.getValue()),
    meta: {
      mono: true,
      severity: (_v: number, row: SensitivityEntry) => statusToSeverity(row.sv_status),
    },
  }),
  col.accessor("normalized_effective_sv", {
    header: () => (
      <HeaderWithTooltip
        label="Normalized SV"
        tooltip="Normalized effective singular value: effective SV after scaling the sensitivity matrix by prior SD (columns) and observation noise SD (rows). Units: prior-SD of parameter change per noise-SD of output change. Thresholds: >10 = pass (data overwhelms prior), >1 = warn (borderline), ≤1 = fail (data uninformative)."
      />
    ),
    cell: (info) => formatSV(info.getValue()),
    meta: {
      mono: true,
      severity: (_v: number, row: SensitivityEntry) => statusToSeverity(row.normalized_sv_status),
    },
  }),
];

export function SensitivityAnalysisTable({
  result,
}: {
  result: SensitivityAnalysisResult;
}) {
  return (
    <div className="space-y-2">
      <Card>
        <CardContent className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 py-3 text-sm">
          <span className="inline-flex items-center gap-1.5 font-medium">
            Sensitivity Analysis
            <StatTooltip explanation="Output sensitivity analysis checks structural identifiability by computing how the emitted-observation moment summary changes with each parameter. Uses the Jacobian of means, same-row covariance entries, and adjacent-row lagged cross-covariance entries evaluated at multiple prior draws, then SVD to detect aliased parameter directions." />
          </span>

          <span className="inline-flex items-center gap-1 text-muted-foreground">
            <span>Deficient directions:</span>
            <span className="tabular-nums text-foreground">
              {result.deficiency_count}/{result.n_parameters}
            </span>
            <StatTooltip explanation="Number of parameter-space directions whose normalized singular value falls below 1 (less than one noise-SD of moment change per prior-SD of parameter change). Zero means all directions are structurally identifiable." />
          </span>

          <Badge variant="secondary">params: {result.n_parameters}</Badge>
          <Badge variant="secondary">draws: {result.n_draws}</Badge>
        </CardContent>
      </Card>

      <InfoTable
        columns={columns as ColumnDef<SensitivityEntry, unknown>[]}
        data={result.per_parameter}
      />
    </div>
  );
}
