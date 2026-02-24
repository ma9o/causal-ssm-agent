import { FunctionalSpecLink } from "@/components/stages/model-spec/functional-spec-link";
import { MeasurementTable } from "@/components/stages/model-spec/measurement-table";
import { PriorTable } from "@/components/stages/model-spec/prior-table";
import { RetryIndicator } from "@/components/stages/model-spec/retry-indicator";
import { SSMEquationDisplay } from "@/components/stages/model-spec/ssm-equation-display";
import type { Extraction, PriorProposal, Stage4Data } from "@causal-ssm/api-types";

/** Extract latent state names from AR coefficient parameters. */
function stateNames(parameters: Stage4Data["model_spec"]["parameters"]): string[] {
  return parameters
    .filter((p) => p.role === "ar_coefficient")
    .map((p) => p.name.split("_").slice(1).join("_"));
}

/**
 * Build synthetic PriorProposal entries for initial state parameters.
 * These match the SSMPriors defaults (t0_means and t0_var_diag).
 */
function initialStatePriors(parameters: Stage4Data["model_spec"]["parameters"]): PriorProposal[] {
  const states = stateNames(parameters);
  const priors: PriorProposal[] = [];
  for (const s of states) {
    priors.push({
      parameter: `t0_mean_${s}`,
      distribution: "Normal",
      params: { mu: 0, sigma: 2 },
      sources: [],
      reasoning: `Default weakly informative prior for the initial state mean of ${s.replace(/_/g, " ")}.`,
    });
    priors.push({
      parameter: `t0_sd_${s}`,
      distribution: "HalfNormal",
      params: { sigma: 2 },
      sources: [],
      reasoning: `Default weakly informative prior for the initial state standard deviation of ${s.replace(/_/g, " ")}.`,
    });
  }
  return priors;
}

export default function Stage4Content({
  data,
  extractions,
}: {
  data: Stage4Data;
  extractions?: Extraction[];
}) {
  const t0Priors = initialStatePriors(data.model_spec.parameters);
  const allPriors = [...data.priors, ...t0Priors];

  return (
    <div className="space-y-4">
      <SSMEquationDisplay
        likelihoods={data.model_spec.likelihoods}
        parameters={data.model_spec.parameters}
        priors={allPriors}
      />
      {extractions && extractions.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Measurement Model</h3>
            <FunctionalSpecLink />
          </div>
          <MeasurementTable
            likelihoods={data.model_spec.likelihoods}
            extractions={extractions}
            priorPredictiveSamples={(data.prior_predictive_samples ?? undefined) as Record<string, number[]> | undefined}
          />
        </div>
      )}
      {allPriors.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Prior Distributions</h3>
          <PriorTable priors={allPriors} parameters={data.model_spec.parameters} />
        </div>
      )}
      {data.validation_retries && data.validation_retries.length > 0 && (
        <RetryIndicator retries={data.validation_retries} />
      )}
    </div>
  );
}
