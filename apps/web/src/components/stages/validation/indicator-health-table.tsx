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
import { useMemo } from "react";

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

function computeColumnSummaries(rows: IndicatorAuditRow[]): Record<StatusField, ColumnIssueSummary> {
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
    col.accessor((row) => row.profile?.n_obs, {
      id: "n_obs",
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Obs"
            tooltip="Number of model-ready observations available for this indicator."
          />
          <IssueBadge summary={summaries.n_obs} />
        </span>
      ),
      cell: (info) => {
        const value = info.getValue();
        return value == null ? "--" : value.toLocaleString();
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(rowStatus(row, "n_obs")),
      },
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
    col.accessor((row) => row.profile?.variance, {
      id: "variance",
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Variance"
            tooltip="Sample variance of the model-ready values. Near-zero variance means the series is effectively constant."
          />
          <IssueBadge summary={summaries.variance} />
        </span>
      ),
      cell: (info) => {
        const value = info.getValue();
        return value == null ? "--" : formatNumber(value);
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(rowStatus(row, "variance")),
      },
    }),
    col.accessor((row) => row.profile?.n_unparseable_timestamps, {
      id: "n_unparseable_timestamps",
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Bad TS"
            tooltip="Count of timestamps that could not be parsed during validation."
          />
          <IssueBadge summary={summaries.n_unparseable_timestamps} />
        </span>
      ),
      cell: (info) => {
        const value = info.getValue();
        return value == null ? "--" : String(value);
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(rowStatus(row, "n_unparseable_timestamps")),
      },
    }),
    col.accessor((row) => row.profile?.time_coverage_ratio, {
      id: "time_coverage_ratio",
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Coverage"
            tooltip="Fraction of the requested time span covered by extracted observations."
          />
          <IssueBadge summary={summaries.time_coverage_ratio} />
        </span>
      ),
      cell: (info) => {
        const value = info.getValue();
        return value == null ? "--" : formatNumber(value);
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(rowStatus(row, "time_coverage_ratio")),
      },
    }),
    col.accessor((row) => row.profile?.max_gap_ratio, {
      id: "max_gap_ratio",
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Max Gap"
            tooltip="Largest timestamp gap relative to the acceptable gap threshold."
          />
          <IssueBadge summary={summaries.max_gap_ratio} />
        </span>
      ),
      cell: (info) => {
        const value = info.getValue();
        return value == null ? "--" : formatNumber(value);
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(rowStatus(row, "max_gap_ratio")),
      },
    }),
    col.accessor((row) => row.profile?.dtype_violations, {
      id: "dtype_violations",
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Type Viol."
            tooltip="Number of values that violated the expected measurement dtype."
          />
          <IssueBadge summary={summaries.dtype_violations} />
        </span>
      ),
      cell: (info) => {
        const value = info.getValue();
        return value == null ? "--" : String(value);
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(rowStatus(row, "dtype_violations")),
      },
    }),
    col.accessor((row) => row.profile?.duplicate_pct, {
      id: "duplicate_pct",
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Dup %"
            tooltip="Share of repeated values that may indicate extraction artifacts."
          />
          <IssueBadge summary={summaries.duplicate_pct} />
        </span>
      ),
      cell: (info) => {
        const value = info.getValue();
        return value == null ? "--" : formatNumber(value);
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(rowStatus(row, "duplicate_pct")),
      },
    }),
    col.accessor((row) => row.profile?.arithmetic_sequence_detected, {
      id: "arithmetic_sequence_detected",
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Arith. Seq."
            tooltip="Whether the indicator values form a suspicious arithmetic sequence."
          />
          <IssueBadge summary={summaries.arithmetic_sequence_detected} />
        </span>
      ),
      cell: (info) =>
        info.getValue() ? "detected" : <span className="text-muted-foreground">none</span>,
      meta: {
        severity: (_v, row) => cellSeverity(rowStatus(row, "arithmetic_sequence_detected")),
      },
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
