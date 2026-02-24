import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { formatNumber } from "@/lib/utils/format";
import type { CellStatus, IndicatorHealth } from "@causal-ssm/api-types";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";

const col = createColumnHelper<IndicatorHealth>();

/** Map a backend cell status to severity. */
function cellSeverity(status: CellStatus | undefined): "fail" | "warn" | undefined {
  if (status === "error") return "fail";
  if (status === "warning") return "warn";
  return undefined;
}

type ColumnIssueSummary = { count: number; hasError: boolean };

const STATUS_FIELDS = [
  "n_obs",
  "variance",
  "time_coverage_ratio",
  "max_gap_ratio",
  "dtype_violations",
  "duplicate_pct",
  "arithmetic_sequence_detected",
] as const;

function computeColumnSummaries(
  rows: IndicatorHealth[],
): Record<string, ColumnIssueSummary> {
  const summaries: Record<string, ColumnIssueSummary> = {};
  for (const field of STATUS_FIELDS) {
    let count = 0;
    let hasError = false;
    for (const row of rows) {
      const status = row.cell_statuses?.[field];
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

function rowIssueCount(row: IndicatorHealth): number {
  let count = 0;
  for (const field of STATUS_FIELDS) {
    const status = row.cell_statuses?.[field];
    if (status === "warning" || status === "error") count++;
  }
  return count;
}

function rowIssueSummary(row: IndicatorHealth): ColumnIssueSummary {
  let count = 0;
  let hasError = false;
  for (const field of STATUS_FIELDS) {
    const status = row.cell_statuses?.[field];
    if (status === "warning") count++;
    if (status === "error") {
      count++;
      hasError = true;
    }
  }
  return { count, hasError };
}

function buildColumns(summaries: Record<string, ColumnIssueSummary>) {
  return [
    col.accessor("indicator", {
      header: "Indicator",
      cell: (info) => <span className="font-medium">{info.getValue()}</span>,
    }),
    col.accessor((row) => rowIssueCount(row), {
      id: "issues",
      header: "Issues",
      cell: ({ row }) => {
        const { count, hasError } = rowIssueSummary(row.original);
        if (count === 0) return <span className="text-muted-foreground">--</span>;
        return (
          <span
            className={
              hasError
                ? "font-semibold text-destructive"
                : "font-semibold text-warning-foreground"
            }
          >
            {count}
          </span>
        );
      },
      meta: { align: "right" },
    }),
    col.accessor("n_obs", {
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Obs"
            tooltip="Number of non-null observations after extraction. More observations generally yield more reliable estimates."
          />
          <IssueBadge summary={summaries.n_obs} />
        </span>
      ),
      cell: (info) => info.getValue().toLocaleString(),
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(row.cell_statuses?.n_obs),
      },
    }),
    col.accessor("variance", {
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Variance"
            tooltip="Sample variance of the indicator values. Near-zero variance means the series is effectively constant and carries no information."
          />
          <IssueBadge summary={summaries.variance} />
        </span>
      ),
      cell: (info) => {
        const v = info.getValue();
        return v === null ? "--" : formatNumber(v);
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(row.cell_statuses?.variance),
      },
    }),
    col.accessor("time_coverage_ratio", {
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Time Coverage"
            tooltip="Fraction of the requested time range that has data. Values close to 1.0 indicate good temporal coverage."
          />
          <IssueBadge summary={summaries.time_coverage_ratio} />
        </span>
      ),
      cell: (info) => {
        const v = info.getValue();
        return v === null ? "--" : formatNumber(v);
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(row.cell_statuses?.time_coverage_ratio),
      },
    }),
    col.accessor("max_gap_ratio", {
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Max Gap"
            tooltip="Longest consecutive gap without data as a fraction of the total time range. Large values indicate periods where the indicator is missing."
          />
          <IssueBadge summary={summaries.max_gap_ratio} />
        </span>
      ),
      cell: (info) => {
        const v = info.getValue();
        return v === null ? "--" : formatNumber(v);
      },
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(row.cell_statuses?.max_gap_ratio),
      },
    }),
    col.accessor("dtype_violations", {
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Dtype Violations"
            tooltip="Number of values that could not be converted to the expected numeric type. Non-zero counts suggest data quality issues at the source."
          />
          <IssueBadge summary={summaries.dtype_violations} />
        </span>
      ),
      cell: (info) => info.getValue(),
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(row.cell_statuses?.dtype_violations),
      },
    }),
    col.accessor("duplicate_pct", {
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Dup %"
            tooltip="Percentage of duplicate timestamp-value pairs. High duplication may indicate redundant data or extraction errors."
          />
          <IssueBadge summary={summaries.duplicate_pct} />
        </span>
      ),
      cell: (info) => formatNumber(info.getValue()),
      meta: {
        align: "right",
        severity: (_v, row) => cellSeverity(row.cell_statuses?.duplicate_pct),
      },
    }),
    col.accessor("arithmetic_sequence_detected", {
      header: () => (
        <span className="inline-flex items-center">
          <HeaderWithTooltip
            label="Arith. Seq."
            tooltip="Whether the values form an arithmetic sequence (constant step between consecutive observations). Detected sequences often indicate synthetic or interpolated data rather than real measurements."
          />
          <IssueBadge summary={summaries.arithmetic_sequence_detected} />
        </span>
      ),
      cell: (info) =>
        info.getValue() ? (
          "detected"
        ) : (
          <span className="text-muted-foreground">none</span>
        ),
      meta: {
        severity: (v: boolean) => (v ? "warn" : undefined),
      },
    }),
  ];
}

export function IndicatorHealthTable({ rows }: { rows: IndicatorHealth[] }) {
  const summaries = useMemo(() => computeColumnSummaries(rows), [rows]);
  const columns = useMemo(() => buildColumns(summaries), [summaries]);
  return <InfoTable columns={columns as ColumnDef<IndicatorHealth, unknown>[]} data={rows} />;
}
