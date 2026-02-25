"use client";

import { TreatmentRankingTable } from "@/components/stages/inference/treatment-ranking-table";
import type { Stage6Data } from "@causal-ssm/api-types";

export default function Stage6Content({ data }: { data: Stage6Data }) {
  if (data.intervention_results.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
        No treatment effects were estimated. This may happen if no treatments passed
        identification checks.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <TreatmentRankingTable results={data.intervention_results} />
    </div>
  );
}
