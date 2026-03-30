"use client";

import { TreatmentRankingTable } from "@/components/stages/inference/treatment-ranking-table";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import type { Stage6Data } from "@causal-ssm/api-types";
import { Bot } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

function getStage6Narrative(data: Stage6Data): string | null {
  const content = data.final_summary?.trim();
  if (content) {
    return content;
  }
  return null;
}

export default function Stage6Content({
  data,
}: {
  data: Stage6Data;
}) {
  const narrative = getStage6Narrative(data);

  return (
    <div className="space-y-4">
      {narrative ? (
        <div className="rounded-lg border bg-muted/20 p-4">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <Bot className="h-3.5 w-3.5" />
            Stage Interpretation
          </div>
          <div className="prose prose-sm max-w-none overflow-y-auto text-sm [&_p]:my-2 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0" style={{ maxHeight: "7.5rem" }}>
            <Markdown remarkPlugins={[remarkGfm]}>{narrative}</Markdown>
          </div>
        </div>
      ) : null}
      {data.intervention_results.length > 0 ? (
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-semibold">Baseline interventional ranking</span>
          <StatTooltip explanation="Total outcome effects ranked under do(treatment = baseline + 1). These are not direct edge coefficients. Peak timing comes from the forward simulation summary, and indicator details are a measurement projection of the outcome effect." />
        </div>
      ) : null}
      {data.intervention_results.length === 0 ? (
        <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
          No treatment effects were estimated. This may happen if no treatments passed
          identification checks.
        </div>
      ) : (
        <TreatmentRankingTable
          results={data.intervention_results}
        />
      )}
    </div>
  );
}
