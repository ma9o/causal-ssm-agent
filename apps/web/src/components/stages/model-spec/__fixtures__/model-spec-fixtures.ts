import type { Stage1bData, Stage4Data } from "@causal-ssm/api-types";
import { collectStage4Priors } from "@/lib/stage4-data";
import stage1bFixture from "../../../../../../../data/DOCTOLIB/run/stage-1b.json";
import stage4Fixture from "../../../../../../../data/DOCTOLIB/run/stage-4.json";

const stage4 = stage4Fixture as unknown as Stage4Data;
const stage1b = stage1bFixture as unknown as Stage1bData;

export const likelihoods = stage4.model_spec.likelihoods;
export const parameters = stage4.model_spec.parameters;
export const priors = collectStage4Priors(stage4);
export const priorPredictiveSamples = stage4.prior_predictive_samples as
  | Record<string, number[]>
  | undefined;

export const indicatorConstructMap: Record<string, string> = Object.fromEntries(
  stage1b.causal_spec.measurement.indicators.map((ind) => [ind.name, ind.construct_name]),
);
