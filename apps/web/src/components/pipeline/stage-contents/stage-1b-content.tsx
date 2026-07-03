"use client";

import { type ConstructStatus, StructureDag } from "@/components/dag/structure-dag";
import { IndicatorTable } from "@/components/stages/measurement/indicator-table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import type { Stage1bData } from "@nof1-causal-lab/api-types";
import { AlertTriangle } from "lucide-react";
import { useMemo } from "react";

function useNodeStatuses(data: Stage1bData): Record<string, ConstructStatus> {
  const spec = data.causal_spec;
  return useMemo(() => {
    const statuses: Record<string, ConstructStatus> = {};

    // Marginalized confounders integrated out across all identified treatments.
    const marginalized = new Set<string>();
    for (const status of Object.values(spec.identifiability?.identifiable_treatments ?? {})) {
      for (const c of status?.marginalized_confounders ?? []) {
        marginalized.add(c);
      }
    }

    // Blocked treatments and their blocking confounders read red on both ends.
    const blocking = new Set<string>();
    for (const [treatment, status] of Object.entries(
      spec.identifiability?.non_identifiable_treatments ?? {},
    )) {
      blocking.add(treatment);
      for (const c of status?.confounders ?? []) {
        blocking.add(c);
      }
    }

    for (const c of spec.latent.constructs) {
      if (blocking.has(c.name)) {
        statuses[c.name] = "blocking";
      } else if (marginalized.has(c.name)) {
        statuses[c.name] = "marginalized";
      } else {
        statuses[c.name] = "observed";
      }
    }

    return statuses;
  }, [spec]);
}

export default function Stage1bContent({ data }: { data: Stage1bData }) {
  const spec = data.causal_spec;
  const nonId = spec.identifiability?.non_identifiable_treatments ?? {};
  const nonIdEntries = Object.entries(nonId);
  const nodeStatuses = useNodeStatuses(data);

  return (
    <div className="space-y-4">
      {nonIdEntries.length > 0 && (
        <Alert variant="warning" className="border-2">
          <AlertTriangle className="h-5 w-5 mt-0.5" />
          <AlertTitle className="text-base font-semibold">
            Some Treatment Effects Were Excluded
          </AlertTitle>
          <AlertDescription className="mt-2 space-y-2">
            <p>
              {nonIdEntries.length} treatment(s) remain non-identifiable and will be excluded from
              downstream intervention analysis. Identifiable treatments still remain.
            </p>
            <div className="space-y-1.5">
              {nonIdEntries.map(([name, status]) => (
                <div key={name} className="flex flex-wrap items-center gap-1.5 text-sm">
                  <span className="font-medium">{name}</span>
                  <span className="text-warning/70">&larr;</span>
                  {status?.confounders.map((c) => (
                    <Badge key={c} variant="warning" className="text-xs">
                      {c}
                    </Badge>
                  ))}
                  {status?.notes && (
                    <span className="text-muted-foreground text-xs">({status.notes})</span>
                  )}
                </div>
              ))}
            </div>
          </AlertDescription>
        </Alert>
      )}
      <StructureDag
        constructs={spec.latent.constructs}
        edges={spec.latent.edges}
        indicators={spec.measurement.indicators}
        nodeStatuses={nodeStatuses}
      />
      <IndicatorTable indicators={spec.measurement.indicators} />
    </div>
  );
}
