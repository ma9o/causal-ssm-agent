import { STAGES, STAGE_IDS } from "@causal-ssm/api-types";
import type { StageMeta } from "@causal-ssm/api-types";

export { STAGES, STAGE_IDS };
export type { StageId, StageMeta } from "@causal-ssm/api-types";

export const FUNCTIONAL_SPEC_URL =
  "https://github.com/ma9o/causal-ssm-agent/blob/master/docs/model-runtime/functional-specification.md#15-parameter-roles-and-constraints";

export function getStageForPrefectRunName(runName: string): StageMeta | undefined {
  let bestMatch: StageMeta | undefined;

  for (const stage of STAGES) {
    if (!runName.startsWith(stage.prefectFlowName)) {
      continue;
    }
    if (!bestMatch || stage.prefectFlowName.length > bestMatch.prefectFlowName.length) {
      bestMatch = stage;
    }
  }

  return bestMatch;
}
