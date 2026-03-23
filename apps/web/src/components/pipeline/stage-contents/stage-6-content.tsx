"use client";

import {
  Stage6AssistantPanel,
  type Stage6AssistantDemoState,
} from "@/components/stages/inference/stage-6-assistant-panel";
import { TreatmentRankingTable } from "@/components/stages/inference/treatment-ranking-table";
import type { Stage6Data } from "@causal-ssm/api-types";

export default function Stage6Content({
  data,
  userId,
  assistantDemoState,
}: {
  data: Stage6Data;
  userId?: string;
  assistantDemoState?: Stage6AssistantDemoState;
}) {
  return (
    <div className="space-y-4">
      {data.intervention_results.length === 0 ? (
        <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
          No treatment effects were estimated. This may happen if no treatments passed
          identification checks.
        </div>
      ) : (
        <TreatmentRankingTable results={data.intervention_results} />
      )}
      {userId ? <Stage6AssistantPanel userId={userId} demoState={assistantDemoState} /> : null}
    </div>
  );
}
