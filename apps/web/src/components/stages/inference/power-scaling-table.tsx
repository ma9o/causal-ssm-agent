import { useMemo } from "react";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import { formatNumber } from "@/lib/utils/format";
import type { PowerScalingResult } from "@causal-ssm/api-types";

const col = createColumnHelper<PowerScalingResult>();

const baseColumns = [
  col.accessor("parameter", {
    header: "Parameter",
    cell: (info) => (
      <span className="font-medium">{info.getValue()}</span>
    ),
    meta: { mono: true },
  }),
  col.accessor("prior_sensitivity", {
    header: () => (
      <HeaderWithTooltip
        label="Prior Sens."
        tooltip="How much the posterior changes when the prior is scaled. Low values (<0.05) are good."
      />
    ),
    cell: (info) => formatNumber(info.getValue()),
    meta: {
      align: "right" as const,
      mono: true,
      severity: (_v: number, row: PowerScalingResult) => {
        if (row.diagnosis === "prior_data_conflict") return "fail";
        if (row.diagnosis === "prior_dominated") return "warn";
        return undefined;
      },
    },
  }),
  col.accessor("likelihood_sensitivity", {
    header: () => (
      <HeaderWithTooltip
        label="Lik. Sens."
        tooltip="How much the posterior changes when the likelihood is scaled. Values > 0.05 suggest data-prior tension if prior sensitivity is also high."
      />
    ),
    cell: (info) => formatNumber(info.getValue()),
    meta: {
      align: "right" as const,
      mono: true,
      severity: (_v: number, row: PowerScalingResult) => {
        if (row.diagnosis === "prior_data_conflict") return "fail";
        return undefined;
      },
    },
  }),
];

const psisColumn = col.accessor("psis_k_hat", {
  header: () => (
    <HeaderWithTooltip
      label="PSIS k"
      tooltip="Pareto k diagnostic for importance sampling reliability. Values > 0.7 indicate unreliable sensitivity estimates."
    />
  ),
  cell: (info) => {
    const v = info.getValue();
    return v == null ? "—" : formatNumber(v, 2);
  },
  meta: {
    align: "right" as const,
    mono: true,
    severity: (v: number | null | undefined) => {
      if (v == null) return undefined;
      if (v > 0.7) return "fail";
      if (v > 0.5) return "warn";
      return undefined;
    },
  },
});

export function PowerScalingTable({ results }: { results: PowerScalingResult[] }) {
  const hasPsis = results.some((p) => p.psis_k_hat != null);

  const columns = useMemo<ColumnDef<PowerScalingResult, unknown>[]>(
    () => (hasPsis ? [...baseColumns, psisColumn] : baseColumns) as ColumnDef<PowerScalingResult, unknown>[],
    [hasPsis],
  );

  return <InfoTable columns={columns} data={results} />;
}
