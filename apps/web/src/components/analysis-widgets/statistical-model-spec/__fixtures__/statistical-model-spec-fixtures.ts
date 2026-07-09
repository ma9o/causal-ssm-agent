import type {
  MeasurementStructureViewData,
  ValidationReportData,
  StatisticalModelSpecData,
} from "@nof1-causal-lab/api-types";
import { collectModelSpecUiPriors } from "@/lib/model-spec-data";
import { buildModelSpecLikelihoodDiagnostics } from "@/lib/model-spec-likelihood-diagnostics";
import { combinedExtractionsSample } from "@/components/__fixtures__/measurements-data";
import {
  demoMeasurementStructure,
  demoStatisticalModelSpec,
  demoValidationReport,
} from "../../../__fixtures__/demo-artifacts";

const validationReport = demoValidationReport as ValidationReportData;
const measurementStructure = demoMeasurementStructure as MeasurementStructureViewData;
const modelSpec = demoStatisticalModelSpec as StatisticalModelSpecData;

// Mirrors the production loader `deriveStatisticalModelSpecData`: likelihood diagnostics are built from the
// validation indicator audits and the extraction observation sample, not recomputed in the story layer.
export const modelSpecData = {
  ...(demoStatisticalModelSpec as object),
  likelihood_diagnostics: buildModelSpecLikelihoodDiagnostics({
    likelihoods: modelSpec.statistical_model_spec.likelihoods,
    indicatorAudits: validationReport.indicators,
    observations: combinedExtractionsSample,
  }),
} as StatisticalModelSpecData;

export const likelihoods = modelSpecData.statistical_model_spec.likelihoods;
export const parameters = modelSpecData.statistical_model_spec.parameters;
export const priors = collectModelSpecUiPriors(modelSpecData);
export const indicators = measurementStructure.causal_design.measurement.indicators;
export const likelihoodDiagnostics = modelSpecData.likelihood_diagnostics;
export const priorPredictiveSamples = modelSpecData.prior_predictive_samples as
  | Record<string, number[]>
  | undefined;
