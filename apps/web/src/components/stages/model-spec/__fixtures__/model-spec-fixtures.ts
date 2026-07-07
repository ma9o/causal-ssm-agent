import type { Stage1bData, Stage3Data, Stage4Data } from "@nof1-causal-lab/api-types";
import { collectStage4UiPriors } from "@/lib/stage4-data";
import { buildStage4LikelihoodDiagnostics } from "@/lib/stage4-likelihood-diagnostics";
import { combinedExtractionsSample } from "@/components/__fixtures__/stage2-data";
import stage1bFixture from "../../../__fixtures__/demo-run/stage-1b.json";
import stage3Fixture from "../../../__fixtures__/demo-run/stage-3.json";
import stage4Fixture from "../../../__fixtures__/demo-run/stage-4.json";

const stage3 = stage3Fixture as unknown as Stage3Data;
const stage1b = stage1bFixture as unknown as Stage1bData;
const stage4 = stage4Fixture as unknown as Stage4Data;

// Mirrors the production loader `deriveStage4Data`: likelihood diagnostics are built from the
// Stage 3 indicator audits and the Stage 2 observation sample, not recomputed in the story layer.
export const stage4Data = {
  ...(stage4Fixture as object),
  likelihood_diagnostics: buildStage4LikelihoodDiagnostics({
    likelihoods: stage4.model_spec.likelihoods,
    indicatorAudits: stage3.indicators,
    observations: combinedExtractionsSample,
  }),
} as Stage4Data;

export const likelihoods = stage4Data.model_spec.likelihoods;
export const parameters = stage4Data.model_spec.parameters;
export const priors = collectStage4UiPriors(stage4Data);
export const indicators = stage1b.causal_spec.measurement.indicators;
export const likelihoodDiagnostics = stage4Data.likelihood_diagnostics;
export const priorPredictiveSamples = stage4Data.prior_predictive_samples as
  | Record<string, number[]>
  | undefined;
