"use client";

import { DiagnosticsAccordion } from "@/components/stages/inference/diagnostics-accordion";
import { MockMethodSwitcher } from "@/components/stages/inference/mock-method-switcher";
import { isMockMode } from "@/lib/api/mock-provider";
import type { Stage5bData } from "@nof1-causal-lab/api-types";
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
    </div>
  );
}
