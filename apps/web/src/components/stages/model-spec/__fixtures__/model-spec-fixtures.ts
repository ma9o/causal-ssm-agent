import type { Stage1bData, Stage2Data, Stage3Data, Stage4Data } from "@causal-ssm/api-types";
import { collectStage4UiPriors } from "@/lib/stage4-data";
import { buildStage4LikelihoodDiagnostics } from "@/lib/stage4-likelihood-diagnostics";
import stage1bFixture from "../../../../../../../data/DOCTOLIB/run/stage-1b.json";
import stage2Fixture from "../../../../../../../data/DOCTOLIB/run/stage-2.json";
import stage3Fixture from "../../../../../../../data/DOCTOLIB/run/stage-3.json";
import stage4Fixture from "../../../../../../../data/DOCTOLIB/run/stage-4.json";

const stage2 = stage2Fixture as unknown as Stage2Data;
const stage3 = stage3Fixture as unknown as Stage3Data;
export const stage4Data = {
  ...(stage4Fixture as object),
  likelihood_diagnostics: buildStage4LikelihoodDiagnostics({
    likelihoods: (stage4Fixture as unknown as Stage4Data).model_spec.likelihoods,
    indicatorAudits: stage3.indicators,
    observations: stage2.combined_extractions_sample,
  }),
} as Stage4Data;
const stage1b = stage1bFixture as unknown as Stage1bData;

export const likelihoods = stage4Data.model_spec.likelihoods;
export const parameters = stage4Data.model_spec.parameters;
export const priors = collectStage4UiPriors(stage4Data);
export const indicators = stage1b.causal_spec.measurement.indicators;
export const likelihoodDiagnostics = stage4Data.likelihood_diagnostics;
export const priorPredictiveSamples = stage4Data.prior_predictive_samples as
  | Record<string, number[]>
  | undefined;
