"use client";

import { DiagnosticsAccordion } from "@/components/stages/inference/diagnostics-accordion";
import type { Stage5aData } from "@causal-ssm/api-types";

export default function Stage5aContent({ data }: { data: Stage5aData }) {
  return (
    <div className="space-y-4">
      <DiagnosticsAccordion
        sviDiagnostics={data.svi_diagnostics}
        posteriorMarginals={data.posterior_marginals}
        posteriorPairs={data.posterior_pairs}
      />
      <div className="rounded-lg bg-muted p-3 text-xs text-muted-foreground">
        SVI Preflight: {data.inference_metadata.method} |{" "}
        {data.inference_metadata.n_samples} samples
      </div>
    </div>
  );
}
