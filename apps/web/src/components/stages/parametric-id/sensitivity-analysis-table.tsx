"use client";

import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { formatNumber } from "@/lib/utils/format";
import type { SensitivityAnalysisResult, SensitivityEntry } from "@causal-ssm/api-types";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";

const col = createColumnHelper<SensitivityEntry>();

const columns = [
  col.accessor("parameter", {
    header: "Parameter",
    cell: (info) => (
      <span className="font-medium">{info.getValue()}</span>
    ),
    meta: { mono: true },
  }),
  col.accessor("sensitivity_norm", {
    header: () => (
      <HeaderWithTooltip
        label="Sensitivity"
        tooltip="Output sensitivity norm: the L2 norm of the Jacobian column for this parameter. Measures how much predicted observations change when this parameter is perturbed. Higher values indicate stronger identifiability from data."
      />
    ),
    cell: (info) => formatNumber(info.getValue(), 4),
    meta: { mono: true },
  }),
  col.accessor("effective_sv", {
    header: () => (
      <HeaderWithTooltip
        label="Effective SV"
        tooltip="Effective singular value: the minimum singular value among SVD directions where this parameter has significant weight. Captures aliasing — two parameters can each have high sensitivity norms but share a near-singular direction, making their individual effects indistinguishable from data."
      />
    ),
    cell: (info) => {
      const v = info.getValue();
      return v < 0.01 ? v.toExponential(1) : formatNumber(v, 4);
    },
    meta: { mono: true },
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
            <StatTooltip explanation="Output sensitivity analysis checks structural identifiability by computing how predicted observations change with each parameter. Uses the Jacobian of the forward model evaluated at multiple prior draws, then SVD to detect aliased parameter directions." />
          </span>

          <span className="inline-flex items-center gap-1 text-muted-foreground">
            <span>Condition:</span>
            <span className="tabular-nums text-foreground">
              {result.condition_number < 1e6
                ? formatNumber(result.condition_number, 1)
                : result.condition_number.toExponential(1)}
            </span>
            <StatTooltip explanation="Condition number of the sensitivity matrix (max/min singular value). Values near 1 indicate excellent identifiability. Very large values (>10⁶) suggest near-singular directions where parameter combinations are aliased." />
          </span>

          <Badge variant="secondary">
            params: {result.n_parameters}
          </Badge>
          <Badge variant="secondary">
            draws: {result.n_draws}
          </Badge>
        </CardContent>
      </Card>

      <InfoTable
        columns={columns as ColumnDef<SensitivityEntry, unknown>[]}
        data={result.per_parameter}
      />
    </div>
  );
}
