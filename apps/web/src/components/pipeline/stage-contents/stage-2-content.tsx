import { DataTable } from "@/components/ui/data-table";
import type { Stage2Data } from "@causal-ssm/api-types";
import { CheckCircle2, XCircle } from "lucide-react";

export default function Stage2Content({ data }: { data: Stage2Data }) {
  if (data.workers.length === 0) {
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
          {Object.values(data.per_indicator_counts).reduce((a, b) => (a ?? 0) + (b ?? 0), 0)?.toLocaleString()} extractions
        </span>
      </div>

      {errors.length > 0 && (
        <div className="space-y-1">
          {errors.map((w) => (
            <p key={w.worker_id} className="text-xs text-destructive">
              Worker {w.worker_id}: {w.error}
            </p>
          ))}
        </div>
      )}

      <DataTable rows={data.combined_extractions_sample} />
    </div>
  );
}
