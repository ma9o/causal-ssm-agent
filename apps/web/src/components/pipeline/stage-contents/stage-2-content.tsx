import { DataTable } from "@/components/ui/data-table";
import { ExploreDataframeButton } from "@/components/ui/explore-dataframe-button";
import type { Stage2Data } from "@causal-ssm/api-types";
import { CheckCircle2, XCircle } from "lucide-react";

export default function Stage2Content({
  data,
  workspaceId,
}: {
  data: Stage2Data;
  workspaceId: string;
}) {
  const totalExtractions = Object.values(data.per_indicator_counts).reduce<number>(
    (sum, count) => sum + (count ?? 0),
    0,
  );

  if (data.workers.length === 0 && totalExtractions === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
        No extraction workers were dispatched. Check if indicators were defined in the previous
        stage.
      </div>
    );
  }

  const succeeded = data.workers.filter((w) => w.status === "completed").length;
  const failed = data.workers.filter((w) => w.status === "failed").length;
  const errors = data.workers.filter((w) => w.status === "failed" && w.error);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-sm">
          <span className="flex items-center gap-1.5 text-success">
            <CheckCircle2 className="h-3.5 w-3.5" />
            {succeeded} succeeded
          </span>
          {failed > 0 && (
            <span className="flex items-center gap-1.5 text-destructive">
              <XCircle className="h-3.5 w-3.5" />
              {failed} failed
            </span>
          )}
          <span className="text-muted-foreground">
            {totalExtractions.toLocaleString()} extractions
          </span>
        </div>
        <ExploreDataframeButton stage="stage-2" workspaceId={workspaceId} />
      </div>

      {errors.length > 0 && (
        <div className="max-h-40 overflow-y-auto space-y-1 rounded-md border border-destructive/30 bg-destructive/5 p-2">
          {errors.map((w) => (
            <p key={w.worker_id} className="text-xs text-destructive">
              Worker {w.worker_id}: {w.error}
            </p>
          ))}
        </div>
      )}

      {data.combined_extractions_sample.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Showing a sample of {data.combined_extractions_sample.length} rows out of{" "}
          {totalExtractions.toLocaleString()}
        </p>
      )}
      <DataTable rows={data.combined_extractions_sample} />
    </div>
  );
}
