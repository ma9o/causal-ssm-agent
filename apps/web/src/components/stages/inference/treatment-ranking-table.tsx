import { useMemo } from "react";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import { formatNumber, formatPercent } from "@/lib/utils/format";
import type { TreatmentEffect } from "@causal-ssm/api-types";

const col = createColumnHelper<TreatmentEffect>();

/** Fail if not identifiable, warn if prior-sensitive. */
function effectSeverity(row: TreatmentEffect): "fail" | "warn" | undefined {
  if (!row.identifiable) return "fail";
  if (row.prior_sensitivity_warning) return "warn";
  return undefined;
}

const columns = [
  col.accessor("treatment", {
    header: "Treatment",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("effect_size", {
    header: () => (
      <HeaderWithTooltip
        label={"\u03C4\u0302"}
        tooltip="Estimated individual treatment effect (ITE) via do-operator steady-state intervention. Positive values indicate the treatment increases the outcome."
      />
    ),
    cell: (info) => {
      const v = info.getValue();
      return v === null ? "—" : formatNumber(v);
    },
    meta: {
      align: "right",
      mono: true,
      severity: (_v: number | null, row: TreatmentEffect) => effectSeverity(row),
    },
  }),
  col.display({
    id: "ci",
    header: () => (
      <HeaderWithTooltip
        label="95% CI"
        tooltip="95% credible interval. The true effect lies within this range with 95% posterior probability."
      />
    ),
    cell: ({ row }) => {
      const ci = row.original.credible_interval;
      return ci ? <span>[{formatNumber(ci[0])}, {formatNumber(ci[1])}]</span> : "—";
    },
    meta: {
      align: "right",
      mono: true,
      severity: (_v: unknown, row: TreatmentEffect) => effectSeverity(row),
    },
  }),
  col.accessor("prob_positive", {
    header: () => (
      <HeaderWithTooltip
        label={`P(\u03C4>0)`}
        tooltip="Posterior probability that the treatment effect is positive. Values near 1 or 0 indicate strong directional evidence."
      />
    ),
    cell: (info) => {
      const p = info.getValue();
      return p == null ? "—" : formatPercent(p);
    },
    meta: {
      align: "right",
      mono: true,
      severity: (_v: number | null, row: TreatmentEffect) => effectSeverity(row),
    },
  }),
];

export function TreatmentRankingTable({ results }: { results: TreatmentEffect[] }) {
  const sorted = useMemo(
    () =>
      [...results].sort(
        (a, b) => Math.abs(b.effect_size ?? 0) - Math.abs(a.effect_size ?? 0),
      ),
    [results],
  );

  return (
    <InfoTable
      columns={columns as ColumnDef<TreatmentEffect, unknown>[]}
      data={sorted}
      rowClassName={(_row, index) => (index === 0 ? "bg-emerald-500/10" : undefined)}
    />
  );
}
