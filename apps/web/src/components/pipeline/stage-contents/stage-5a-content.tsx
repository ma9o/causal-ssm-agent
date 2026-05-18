"use client";

import { DiagnosticsAccordion } from "@/components/stages/inference/diagnostics-accordion";
import type { Stage5aData } from "@nof1-causal-lab/api-types";

export default function Stage5aContent({ data }: { data: Stage5aData }) {
  return (
    <div className="space-y-4">
      <DiagnosticsAccordion
        sviDiagnostics={data.svi_diagnostics}
        posteriorMarginals={data.posterior_marginals}
        posteriorPairs={data.posterior_pairs}
      />
    </div>
  );
}
