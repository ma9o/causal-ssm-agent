import { DataSummaryStats } from "@/components/stages/preprocess/data-summary-stats";
import { DataTable } from "@/components/ui/data-table";
import type { Stage0Data } from "@causal-ssm/api-types";

export default function Stage0Content({ data }: { data: Stage0Data }) {
  return (
    <div className="space-y-2">
      <DataSummaryStats
        nRecords={data.n_records}
        nColumns={data.n_columns}
        dateRange={data.date_range}
      />
      {data.column_descriptions.length > 0 && (
        <DataTable
          rows={data.column_descriptions.map((col) => ({
            column: col.name,
            type: col.dtype,
            description: col.description,
          }))}
          maxHeight="max-h-48"
        />
      )}
      {data.sample.length > 0 && <DataTable rows={data.sample} maxHeight="max-h-64" />}
    </div>
  );
}
