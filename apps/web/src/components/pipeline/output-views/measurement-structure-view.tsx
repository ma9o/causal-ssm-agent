"use client";

import { deriveConstructStatuses } from "@/components/dag/construct-statuses";
import { StructureDag } from "@/components/dag/structure-dag";
import { IndicatorTable } from "@/components/analysis-widgets/measurement-structure/indicator-table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import type { MeasurementStructureViewData } from "@nof1-causal-lab/api-types";
import { AlertTriangle } from "lucide-react";

export default function MeasurementStructureView({ data }: { data: MeasurementStructureViewData }) {
  const spec = data.causal_design;
  const nonId = spec.identifiability?.non_identifiable_treatments ?? {};
  const nonIdEntries = Object.entries(nonId);
  const nodeStatuses = deriveConstructStatuses(spec, data.structural_plan);

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
        knownInputs={spec.known_inputs}
        nodeStatuses={nodeStatuses}
      />
      <IndicatorTable indicators={spec.measurement.indicators} />
    </div>
  );
}
