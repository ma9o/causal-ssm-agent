import { Badge } from "@/components/ui/badge";
import { InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import type { Indicator } from "@causal-ssm/api-types";

const col = createColumnHelper<Indicator>();

const columns = [
  col.accessor("name", {
    header: "Name",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("measurement_dtype", {
    header: "Dtype",
    cell: (info) => <Badge variant="outline">{info.getValue()}</Badge>,
  }),
  col.accessor("aggregation", {
    header: "Aggregation",
    cell: (info) => <Badge variant="secondary">{info.getValue()}</Badge>,
  }),
  col.accessor("how_to_measure", {
    header: "How to Measure",
    cell: (info) => (
      <span className="max-w-xs text-muted-foreground">{info.getValue()}</span>
    ),
  }),
];

export function IndicatorTable({ indicators }: { indicators: Indicator[] }) {
  return (
    <InfoTable
      columns={columns as ColumnDef<Indicator, unknown>[]}
      data={indicators}
      groupBy={(row) => row.construct_name}
      renderGroupHeader={(construct, rows) => (
        <>
          <span className="text-sm font-semibold">{construct}</span>
          <span className="ml-2 text-xs text-muted-foreground">
            {rows.length} indicator{rows.length !== 1 && "s"}
          </span>
        </>
      )}
    />
  );
}
