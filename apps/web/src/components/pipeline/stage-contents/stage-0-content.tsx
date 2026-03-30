import { DataSummaryStats } from "@/components/stages/preprocess/data-summary-stats";
import { DataTable } from "@/components/ui/data-table";
import { ExploreDataframeButton } from "@/components/ui/explore-dataframe-button";
import type { Stage0Data } from "@causal-ssm/api-types";
import { useMemo } from "react";

export default function Stage0Content({
  data,
  workspaceId,
}: {
  data: Stage0Data;
  workspaceId: string;
}) {
  const columnTooltips = useMemo(() => {
    const tips: Record<string, string> = {};
    for (const col of data.column_descriptions ?? []) {
      const parts = [col.dtype, col.description].filter(Boolean);
      if (parts.length > 0) tips[col.name] = parts.join(" — ");
    }
    return tips;
  }, [data.column_descriptions]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <DataSummaryStats
          nRecords={data.n_records}
          nColumns={data.n_columns}
          dateRange={data.date_range}
        />
        <ExploreDataframeButton stage="stage-0" workspaceId={workspaceId} />
      </div>
      {data.sample.length > 0 && (
        <>
          <p className="text-xs text-muted-foreground">
            Showing a sample of {data.sample.length} rows out of {data.n_records.toLocaleString()}
          </p>
          <DataTable rows={data.sample} maxHeight="max-h-64" columnTooltips={columnTooltips} />
        </>
      )}
    </div>
  );
}
