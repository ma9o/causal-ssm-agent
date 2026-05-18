import { FunctionalSpecLink } from "@/components/stages/model-spec/functional-spec-link";
import { MeasurementTable } from "@/components/stages/model-spec/measurement-table";
import { PriorTable } from "@/components/stages/model-spec/prior-table";
import { SSMEquationDisplay } from "@/components/stages/model-spec/ssm-equation-display";
import { collectStage4UiPriors } from "@/lib/stage4-data";
import type { Indicator, Stage4Data } from "@nof1-causal-lab/api-types";

export default function Stage4Content({
  data,
  indicators,
}: {
  data: Stage4Data;
  indicators?: Indicator[];
}) {
  const authoredPriors = collectStage4UiPriors(data);
  const hasLikelihoodDiagnostics = Object.values(data.likelihood_diagnostics).some(
    (diagnostics) => (diagnostics?.histogram.length ?? 0) > 0,
  );

  return (
    <div className="space-y-4">
      <SSMEquationDisplay
        likelihoods={data.model_spec.likelihoods}
        parameters={data.model_spec.parameters}
        priors={authoredPriors}
        indicators={indicators}
      />
      {hasLikelihoodDiagnostics && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Likelihoods Diagnostics</h3>
            <FunctionalSpecLink />
          </div>
          <MeasurementTable
            likelihoods={data.model_spec.likelihoods}
            diagnostics={data.likelihood_diagnostics}
            priorPredictiveSamples={
              (data.prior_predictive_samples ?? undefined) as Record<string, number[]> | undefined
            }
          />
        </div>
      )}
      {authoredPriors.length > 0 && (
        <div className="space-y-3">
          <div className="space-y-1">
            <h3 className="text-sm font-semibold">Authored Priors</h3>
            <p className="text-sm text-muted-foreground">
              Only priors explicitly authored in the Stage 4 discrete-time view are shown here.
              Terms without an authored prior are labeled as not authored in the semantic panels.
            </p>
          </div>
          <PriorTable priors={authoredPriors} parameters={data.model_spec.parameters} />
        </div>
      )}
    </div>
  );
}
