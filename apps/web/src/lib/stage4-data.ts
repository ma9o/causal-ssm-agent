import type { PriorProposal, Stage4Data } from "@causal-ssm/api-types";

export function collectStage4Priors(data: Stage4Data): PriorProposal[] {
  return data.resolved_priors.filter((prior): prior is PriorProposal => prior != null);
}

export function collectStage4UiPriors(data: Stage4Data): PriorProposal[] {
  return data.model_spec.parameters.flatMap((parameter) => {
    const prior = data.authored_priors[parameter.name];
    return prior ? [prior] : [];
  });
}
