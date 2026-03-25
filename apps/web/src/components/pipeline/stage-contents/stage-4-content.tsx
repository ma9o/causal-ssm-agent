import { FunctionalSpecLink } from "@/components/stages/model-spec/functional-spec-link";
import { MeasurementTable } from "@/components/stages/model-spec/measurement-table";
import { PriorTable } from "@/components/stages/model-spec/prior-table";
import { SSMEquationDisplay } from "@/components/stages/model-spec/ssm-equation-display";
import { collectStage4Priors } from "@/lib/stage4-data";
import type { Indicator, ObservationRecord, Stage4Data } from "@causal-ssm/api-types";

export default function Stage4Content({
  data,
  extractions,
  indicators,
}: {
  data: Stage4Data;
  extractions?: ObservationRecord[];
  indicators?: Indicator[];
}) {
  const allPriors = collectStage4Priors(data);

  // Build indicator → construct mapping for observation model equations
  const indicatorConstructMap = indicators
    ? Object.fromEntries(indicators.map((ind) => [ind.name, ind.construct_name]))
    : undefined;

  return (
    <div className="space-y-4">
      <SSMEquationDisplay
        likelihoods={data.model_spec.likelihoods}
        parameters={data.model_spec.parameters}
        priors={allPriors}
        indicatorConstructMap={indicatorConstructMap}
      />
      {extractions && extractions.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Likelihoods Diagnostics</h3>
            <FunctionalSpecLink />
          </div>
          <MeasurementTable
            likelihoods={data.model_spec.likelihoods}
            extractions={extractions}
            priorPredictiveSamples={
              (data.prior_predictive_samples ?? undefined) as Record<string, number[]> | undefined
            }
          />
        </div>
      )}
      {allPriors.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Priors Diagnostics</h3>
          <PriorTable priors={allPriors} parameters={data.model_spec.parameters} />
        </div>
      )}
    </div>
  );
}
