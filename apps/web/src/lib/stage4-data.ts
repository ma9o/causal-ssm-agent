import type { PriorProposal, Stage4Data } from "@causal-ssm/api-types";

export function collectStage4Priors(data: Stage4Data): PriorProposal[] {
  return data.resolved_priors.filter((prior): prior is PriorProposal => prior != null);
}
