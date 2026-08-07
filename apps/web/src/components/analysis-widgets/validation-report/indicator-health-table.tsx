import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { formatNumber } from "@/lib/utils/format";
import type {
  CellStatus,
  IndicatorAudit,
  IndicatorEmpiricalProfile,
  IndicatorValidation,
} from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { type ReactNode, useMemo } from "react";

type IndicatorAuditRow = {
  indicator: string;
  profile: IndicatorEmpiricalProfile | null | undefined;
  validation: IndicatorValidation;
};

const col = createColumnHelper<IndicatorAuditRow>();

function cellSeverity(status: CellStatus | undefined): "fail" | "warn" | undefined {
  if (status === "error") return "fail";
  if (status === "warning") return "warn";
  return undefined;
}

type ColumnIssueSummary = { count: number; hasError: boolean };
type StatusField =
  | "n_obs"
  | "variance"
  | "n_unparseable_timestamps"
  | "time_coverage_ratio"
  | "max_gap_ratio"
  | "dtype_violations"
  | "duplicate_pct"
  | "arithmetic_sequence_detected";

const STATUS_FIELDS: StatusField[] = [
  "n_obs",
  "variance",
  "n_unparseable_timestamps",
  "time_coverage_ratio",
  "max_gap_ratio",
  "dtype_violations",
  "duplicate_pct",
  "arithmetic_sequence_detected",
];

function rowStatus(row: IndicatorAuditRow, field: StatusField): CellStatus | undefined {
  return row.validation.checks?.[field];
}

function computeColumnSummaries(
  rows: IndicatorAuditRow[],
): Record<StatusField, ColumnIssueSummary> {
  const summaries = {} as Record<StatusField, ColumnIssueSummary>;
  for (const field of STATUS_FIELDS) {
    let count = 0;
    let hasError = false;
    for (const row of rows) {
      const status = rowStatus(row, field);
      if (status === "warning") count++;
      if (status === "error") {
        count++;
        hasError = true;
      }
    }
    summaries[field] = { count, hasError };
  }
  return summaries;
}

function IssueBadge({ summary }: { summary: ColumnIssueSummary | undefined }) {
  if (!summary || summary.count === 0) return null;
  return (
    <Badge
      variant={summary.hasError ? "destructive" : "warning"}
      className="ml-1 px-1.5 py-0 text-[10px] leading-4"
    >
      {summary.count}
    </Badge>
  );
}

export function summarizeValidationIssues(validation: IndicatorValidation): ColumnIssueSummary {
  let count = 0;
  let hasError = false;
  for (const issue of validation.issues ?? []) {
    if (issue.severity === "info") continue;
    count++;
    if (issue.severity === "error") {
      hasError = true;
    }
  }
  return { count, hasError };
}

function rowIssueSummary(row: IndicatorAuditRow): ColumnIssueSummary {
  return summarizeValidationIssues(row.validation);
}

function buildRows(audits: Record<string, IndicatorAudit | undefined>): IndicatorAuditRow[] {
  return Object.entries(audits)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([indicator, audit]) => ({
      indicator,
      profile: audit?.profile,
      validation: audit?.validation ?? { issues: [], checks: {} },
    }));
}

function buildStatusColumn<TValue extends number | boolean>(
  summaries: Record<StatusField, ColumnIssueSummary>,
  config: {
    field: StatusField;
    label: string;
    tooltip: string;
    value: (row: IndicatorAuditRow) => TValue | null | undefined;
    format: (value: TValue) => ReactNode;
    align?: "right";
  },
) {
  return col.accessor(config.value, {
    id: config.field,
    header: () => (
      <span className="inline-flex items-center">
        <HeaderWithTooltip label={config.label} tooltip={config.tooltip} />
        <IssueBadge summary={summaries[config.field]} />
      </span>
    ),
    cell: (info) => {
      const value = info.getValue();
      return value == null ? "--" : config.format(value);
    },
    meta: {
      align: config.align,
      severity: (_value, row) => cellSeverity(rowStatus(row, config.field)),
    },
  });
}

function buildColumns(summaries: Record<StatusField, ColumnIssueSummary>) {
  return [
    col.accessor("indicator", {
      header: "Indicator",
      cell: (info) => <span className="font-medium">{info.getValue()}</span>,
    }),
    col.accessor((row) => rowIssueSummary(row).count, {
      id: "issues",
      header: "Issues",
      cell: ({ row }) => {
        const { count, hasError } = rowIssueSummary(row.original);
        if (count === 0) return <span className="text-muted-foreground">--</span>;
        return (
          <span
            className={
              hasError ? "font-semibold text-destructive" : "font-semibold text-warning-foreground"
            }
          >
            {count}
          </span>
        );
      },
      meta: { align: "right" },
    }),
    buildStatusColumn(summaries, {
      field: "n_obs",
      label: "Obs",
      tooltip: "Number of model-ready observations available for this indicator.",
      value: (row) => row.profile?.n_obs,
      format: (value) => value.toLocaleString(),
      align: "right",
    }),
    col.accessor((row) => row.profile?.mean, {
      id: "mean",
      header: () => (
        <HeaderWithTooltip
          label="Mean"
          tooltip="Average of the model-ready numeric values. Useful for judging the observed scale."
        />
      ),
      cell: (info) => {
        const value = info.getValue();
        return value == null ? "--" : formatNumber(value);
      },
      meta: { align: "right" },
    }),
    buildStatusColumn(summaries, {
      field: "variance",
      label: "Variance",
      tooltip:
        "Sample variance of the model-ready values. Near-zero variance means the series is effectively constant.",
      value: (row) => row.profile?.variance,
      format: (value) => formatNumber(value),
      align: "right",
    }),
    buildStatusColumn(summaries, {
      field: "n_unparseable_timestamps",
      label: "Bad TS",
      tooltip: "Count of timestamps that could not be parsed during validation.",
      value: (row) => row.profile?.n_unparseable_timestamps,
      format: String,
      align: "right",
    }),
    buildStatusColumn(summaries, {
      field: "time_coverage_ratio",
      label: "Coverage",
      tooltip: "Fraction of the requested time span covered by extracted observations.",
      value: (row) => row.profile?.time_coverage_ratio,
      format: (value) => formatNumber(value),
      align: "right",
    }),
    buildStatusColumn(summaries, {
      field: "max_gap_ratio",
      label: "Max Gap",
      tooltip: "Largest timestamp gap relative to the acceptable gap threshold.",
      value: (row) => row.profile?.max_gap_ratio,
      format: (value) => formatNumber(value),
      align: "right",
    }),
    buildStatusColumn(summaries, {
      field: "dtype_violations",
      label: "Type Viol.",
      tooltip: "Number of values that violated the expected measurement dtype.",
      value: (row) => row.profile?.dtype_violations,
      format: String,
      align: "right",
    }),
    buildStatusColumn(summaries, {
      field: "duplicate_pct",
      label: "Dup %",
      tooltip: "Share of repeated values that may indicate extraction artifacts.",
      value: (row) => row.profile?.duplicate_pct,
      format: (value) => formatNumber(value),
      align: "right",
    }),
    buildStatusColumn(summaries, {
      field: "arithmetic_sequence_detected",
      label: "Arith. Seq.",
      tooltip: "Whether the indicator values form a suspicious arithmetic sequence.",
      value: (row) => row.profile?.arithmetic_sequence_detected,
      format: (value) => (value ? "detected" : <span className="text-muted-foreground">none</span>),
    }),
  ];
}

export function IndicatorHealthTable({
  audits,
}: {
  audits: Record<string, IndicatorAudit | undefined>;
}) {
  const rows = useMemo(() => buildRows(audits), [audits]);
  const summaries = useMemo(() => computeColumnSummaries(rows), [rows]);
  const columns = useMemo(() => buildColumns(summaries), [summaries]);
  return <InfoTable columns={columns as ColumnDef<IndicatorAuditRow, unknown>[]} data={rows} />;
}
