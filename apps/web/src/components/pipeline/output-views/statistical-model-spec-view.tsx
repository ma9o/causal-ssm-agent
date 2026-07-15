import { FunctionalSpecLink } from "@/components/analysis-widgets/statistical-model-spec/functional-spec-link";
import { MeasurementTable } from "@/components/analysis-widgets/statistical-model-spec/measurement-table";
import { PriorTable } from "@/components/analysis-widgets/statistical-model-spec/prior-table";
import { SSMEquationDisplay } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";
import { collectModelSpecUiPriors } from "@/lib/model-spec-data";
import type {
  Indicator,
  PriorPredictiveDiagnostic,
  StatisticalModelSpecData,
} from "@nof1-causal-lab/api-types";

function PriorPredictiveDiagnostics({ diagnostics }: { diagnostics: PriorPredictiveDiagnostic[] }) {
  if (diagnostics.length === 0) return null;
  return (
    <div className="space-y-3">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold">Prior-predictive reachability</h3>
        <p className="text-sm text-muted-foreground">
          Persisted checks from exact construct admission, including feedback-component rechecks.
        </p>
      </div>
      <ul className="grid gap-2 md:grid-cols-2">
        {diagnostics.map((diagnostic, index) => (
          <li
            key={`${diagnostic.check}:${diagnostic.target}:${index}`}
            className={
              diagnostic.passed
                ? "rounded-md border border-success/25 bg-success/5 p-3"
                : "rounded-md border border-warning/30 bg-warning/10 p-3"
            }
          >
            <div className="flex items-center justify-between gap-2 text-xs">
              <span className="font-medium">{diagnostic.check}</span>
              <span className="text-muted-foreground">{diagnostic.passed ? "PASS" : "REVIEW"}</span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {diagnostic.target}: {diagnostic.value}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">Target: {diagnostic.band}</p>
            {!diagnostic.passed && diagnostic.diagnosis.length > 0 && (
              <div className="mt-2 space-y-1 text-xs text-muted-foreground">
                {diagnostic.diagnosis.map((line) => (
                  <p key={line}>{line}</p>
                ))}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function StatisticalModelSpecView({
  data,
  indicators,
}: {
  data: StatisticalModelSpecData;
  indicators?: Indicator[];
}) {
  const authoredPriors = collectModelSpecUiPriors(data);
  const hasLikelihoodDiagnostics = Object.values(data.likelihood_diagnostics).some(
    (diagnostics) => (diagnostics?.histogram.length ?? 0) > 0,
  );

  return (
    <div className="space-y-4">
      <SSMEquationDisplay
        likelihoods={data.statistical_model_spec.likelihoods}
        parameters={data.statistical_model_spec.parameters}
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
            likelihoods={data.statistical_model_spec.likelihoods}
            diagnostics={data.likelihood_diagnostics}
            priorPredictiveSamples={
              (data.prior_predictive_samples ?? undefined) as Record<string, number[]> | undefined
            }
          />
        </div>
      )}
      <PriorPredictiveDiagnostics diagnostics={data.prior_predictive_diagnostics ?? []} />
      {authoredPriors.length > 0 && (
        <div className="space-y-3">
          <div className="space-y-1">
            <h3 className="text-sm font-semibold">Authored Priors</h3>
            <p className="text-sm text-muted-foreground">
              Only priors explicitly authored in the model-spec discrete-time view are shown here.
              Terms without an authored prior are labeled as not authored in the semantic panels.
            </p>
          </div>
          <PriorTable priors={authoredPriors} parameters={data.statistical_model_spec.parameters} />
        </div>
      )}
    </div>
  );
}
