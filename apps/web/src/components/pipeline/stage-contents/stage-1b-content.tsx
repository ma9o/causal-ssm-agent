"use client";

import { CausalDag } from "@/components/dag/causal-dag";
import type { ConstructStatus } from "@/components/dag/construct-node";
import { IndicatorTable } from "@/components/stages/measurement/indicator-table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import type { Stage1bData } from "@nof1-causal-lab/api-types";
import { AlertTriangle } from "lucide-react";
import { useMemo } from "react";

function edgeKey(source: string, target: string): string {
  return `${source} ${target}`;
}

interface IdentifiabilityViewModel {
  nodeStatuses: Record<string, ConstructStatus>;
  blockingEdges: Set<string>;
}

function useIdentifiabilityViewModel(data: Stage1bData): IdentifiabilityViewModel {
  const spec = data.causal_spec;
  return useMemo(() => {
    const statuses: Record<string, ConstructStatus> = {};

    // Collect marginalized confounders across all identified treatments
    const marginalized = new Set<string>();
    for (const status of Object.values(spec.identifiability?.identifiable_treatments ?? {})) {
      for (const c of status?.marginalized_confounders ?? []) {
        marginalized.add(c);
      }
    }

    // Collect both the blocked treatments and their blocking confounders
    // so the non-identifiable pair reads as red on both ends.
    const blocking = new Set<string>();
    const blockingEdges = new Set<string>();
    for (const [treatment, status] of Object.entries(
      spec.identifiability?.non_identifiable_treatments ?? {},
    )) {
      blocking.add(treatment);
      for (const c of status?.confounders ?? []) {
        blocking.add(c);
        blockingEdges.add(edgeKey(c, treatment));
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

    return { nodeStatuses: statuses, blockingEdges };
  }, [spec]);
}

export default function Stage1bContent({ data }: { data: Stage1bData }) {
  const spec = data.causal_spec;
  const nonId = spec.identifiability?.non_identifiable_treatments ?? {};
  const nonIdEntries = Object.entries(nonId);
  const { nodeStatuses, blockingEdges } = useIdentifiabilityViewModel(data);

  return (
    <div className="space-y-4">
      {nonIdEntries.length > 0 &&
        (data.outcome === "fail" ? (
          <Alert variant="destructive" className="border-2">
            <AlertTriangle className="h-5 w-5 mt-0.5" />
            <AlertTitle className="text-base font-semibold">
              Non Identifiable Treatments Detected
            </AlertTitle>
            <AlertDescription className="mt-2 space-y-2">
              <p>
                {nonIdEntries.length} treatment(s) were found to be non-identifiable. If possible, address the blocking confounders (marked with red badges below) to achieve identifiability.
              </p>
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
            </AlertDescription>
          </Alert>
        ) : (
          <Alert variant="warning" className="border-2">
            <AlertTriangle className="h-5 w-5 mt-0.5" />
            <AlertTitle className="text-base font-semibold">
              Some Treatment Effects Were Excluded
            </AlertTitle>
            <AlertDescription className="mt-2 space-y-2">
              <p>
                {nonIdEntries.length} treatment(s) remain non-identifiable and will be excluded
                from downstream intervention analysis. Identifiable treatments still remain.
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
        ))}
      <CausalDag
        constructs={spec.latent.constructs}
        edges={spec.latent.edges}
        indicators={spec.measurement.indicators}
        nodeStatuses={nodeStatuses}
        blockingEdges={blockingEdges}
        height="min(600px, 70vh)"
      />
      <IndicatorTable indicators={spec.measurement.indicators} />
    </div>
  );
}
