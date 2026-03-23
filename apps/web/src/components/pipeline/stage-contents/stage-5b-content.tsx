"use client";

import { DiagnosticsAccordion } from "@/components/stages/inference/diagnostics-accordion";
import { MockMethodSwitcher } from "@/components/stages/inference/mock-method-switcher";
import { isMockMode } from "@/lib/api/mock-provider";
import type { Stage5bData } from "@causal-ssm/api-types";
import { useState } from "react";

export default function Stage5bContent({ workspaceId, data }: { workspaceId: string; data: Stage5bData }) {
  const [activeData, setActiveData] = useState(data);
  const mock = isMockMode();

  return (
    <div className="space-y-4">
      {mock && <MockMethodSwitcher workspaceId={workspaceId} baseData={data} onDataChange={setActiveData} />}
      <DiagnosticsAccordion
        powerScaling={activeData.power_scaling}
        ppc={activeData.ppc}
        mcmcDiagnostics={activeData.mcmc_diagnostics}
        sviDiagnostics={activeData.svi_diagnostics}
        smcDiagnostics={activeData.smc_diagnostics}
        looDiagnostics={activeData.loo_diagnostics}
        posteriorMarginals={activeData.posterior_marginals}
        posteriorPairs={activeData.posterior_pairs}
      />
      <div className="rounded-lg bg-muted p-3 text-xs text-muted-foreground">
        Inference: {activeData.inference_metadata.method} |{" "}
        {activeData.inference_metadata.n_samples} samples |{" "}
        {activeData.inference_metadata.duration_seconds.toFixed(1)}s
      </div>
    </div>
  );
}
