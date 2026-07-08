"use client";

import { DiagnosticsAccordion } from "@/components/analysis-widgets/posterior/diagnostics-accordion";
import { MockMethodSwitcher } from "@/components/analysis-widgets/posterior/mock-method-switcher";
import { isMockMode } from "@/lib/api/mock-provider";
import type { PosteriorData } from "@nof1-causal-lab/api-types";
import { useState } from "react";

export default function PosteriorView({
  workspaceId,
  data,
}: {
  workspaceId: string;
  data: PosteriorData;
}) {
  const [activeData, setActiveData] = useState(data);
  const mock = isMockMode();

  return (
    <div className="space-y-4">
      {mock && (
        <MockMethodSwitcher
          workspaceId={workspaceId}
          baseData={data}
          onDataChange={setActiveData}
        />
      )}
      <DiagnosticsAccordion
        ppc={activeData.ppc}
        mcmcDiagnostics={activeData.mcmc_diagnostics}
        smcDiagnostics={activeData.smc_diagnostics}
        looDiagnostics={activeData.loo_diagnostics}
        posteriorMarginals={activeData.posterior_marginals}
        posteriorPairs={activeData.posterior_pairs}
      />
    </div>
  );
}
