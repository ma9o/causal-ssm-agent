"use client";

import { CausalDag } from "@/components/dag/causal-dag";
import { IndicatorTable } from "@/components/stages/measurement/indicator-table";
import { Badge } from "@/components/ui/badge";
import { HardGateAlert } from "@/components/ui/custom/hard-gate-alert";
import type { Stage1bData } from "@causal-ssm/api-types";

export default function Stage1bContent({ data }: { data: Stage1bData }) {
  const spec = data.causal_spec;
  const nonId = spec.identifiability?.non_identifiable_treatments ?? {};
  const nonIdEntries = Object.entries(nonId);
  const hasNonIdentifiable = nonIdEntries.length > 0;

  return (
    <div className="space-y-4">
      {hasNonIdentifiable && (
        <HardGateAlert
          title="Non-Identifiable Treatment Effects Removed"
          explanation={`${nonIdEntries.length} treatment(s) have non-identifiable causal effects and were filtered out.`}
          suggestion="Add instrumental variables or remove unobserved confounders to enable identification."
        >
          <div className="space-y-1.5">
            {nonIdEntries.map(([name, status]) => (
              <div key={name} className="flex flex-wrap items-center gap-1.5 text-sm">
                <span className="font-medium">{name}</span>
                <span className="text-destructive/70">&larr;</span>
                {status?.confounders.map((c) => (
                  <Badge key={c} variant="destructive" className="text-xs">
                    {c}
                  </Badge>
                ))}
                {status?.notes && (
                  <span className="text-muted-foreground text-xs">({status.notes})</span>
                )}
              </div>
            ))}
          </div>
        </HardGateAlert>
      )}
      <CausalDag
        constructs={spec.latent.constructs}
        edges={spec.latent.edges}
        indicators={spec.measurement.indicators}
        identifiability={spec.identifiability}
        height="min(600px, 70vh)"
      />
      <IndicatorTable indicators={spec.measurement.indicators} />
    </div>
  );
}
