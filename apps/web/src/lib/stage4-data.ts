import {
  OBSERVATION_HYPERPARAMETERS_BY_DISTRIBUTION,
  type Indicator,
  type LikelihoodSpec,
  type ParameterSpec,
  type PriorProposal,
  type Stage4Data,
} from "@nof1-causal-lab/api-types";

export function collectStage4UiPriors(data: Stage4Data): PriorProposal[] {
  return data.statistical_model_spec.parameters.flatMap((parameter) => {
    const prior = data.authored_priors[parameter.name];
    return prior ? [prior] : [];
  });
}

export interface Stage4ObservationPriorTerm {
  parameterName: string;
  prior?: PriorProposal;
}

function orderedThresholdGapsAreActive(indicator?: Indicator): boolean {
  if (!indicator?.ordinal_levels) {
    return true;
  }
  return indicator.ordinal_levels.length > 2;
}

export function collectStage4ObservationPriorTerms({
  likelihood,
  parameters,
  priors,
  indicators,
}: {
  likelihood: LikelihoodSpec;
  parameters: ParameterSpec[];
  priors: PriorProposal[];
  indicators?: Indicator[];
}): Stage4ObservationPriorTerm[] {
  const indicator = indicators?.find((item) => item.name === likelihood.variable);
  const declaredParameters = new Set(parameters.map((parameter) => parameter.name));
  const priorByParameter = new Map(priors.map((prior) => [prior.parameter, prior]));
  const expectedParameterNames: string[] = [];

  if (indicator?.construct_name) {
    const loadingParameterName = `lambda_${likelihood.variable}_${indicator.construct_name}`;
    if (declaredParameters.has(loadingParameterName)) {
      expectedParameterNames.push(loadingParameterName);
    }
  }

  const measurementErrorParameterName = `obs_sd_${likelihood.variable}`;
  if (declaredParameters.has(measurementErrorParameterName)) {
    expectedParameterNames.push(measurementErrorParameterName);
  }

  const observationHyperparameters =
    OBSERVATION_HYPERPARAMETERS_BY_DISTRIBUTION[likelihood.distribution] ?? [];
  for (const parameterName of observationHyperparameters) {
    if (parameterName === "obs_ordered_gaps" && !orderedThresholdGapsAreActive(indicator)) {
      continue;
    }
    if (declaredParameters.has(parameterName)) {
      expectedParameterNames.push(parameterName);
    }
  }

  return expectedParameterNames.map((parameterName) => ({
    parameterName,
    prior: priorByParameter.get(parameterName),
  }));
}
