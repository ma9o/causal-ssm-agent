import { DataSummaryStats } from "@/components/stages/preprocess/data-summary-stats";
import { DataTable } from "@/components/ui/data-table";
import type { Stage0Data } from "@causal-ssm/api-types";

export default function Stage0Content({ data }: { data: Stage0Data }) {
  return (
    <div className="space-y-4">
      <DataSummaryStats
        sourceLabel={data.source_label}
        nRecords={data.n_records}
        dateRange={data.date_range}
      />
      {data.sample.length > 0 && <DataTable rows={data.sample} maxHeight="max-h-64" />}
    </div>
  );
}
