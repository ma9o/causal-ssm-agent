import type {
  BaselineReportData,
  LatentStructureData,
  MeasurementsData,
  MeasurementStructureViewData,
  PosteriorData,
  RawDataData,
  StatisticalModelSpecData,
  ValidationReportData,
} from "@nof1-causal-lab/api-types";
import { demoArtifactSources } from "./demo-artifact-sources";

export const demoRawData = demoArtifactSources.raw_data as RawDataData;
export const demoLatentStructure = demoArtifactSources.latent_structure as LatentStructureData;
export const demoMeasurementStructure = {
  ...demoArtifactSources.measurement_structure,
  causal_design: demoArtifactSources.causal_design,
  structural_plan: demoArtifactSources.structural_plan,
} as unknown as MeasurementStructureViewData;
export const demoMeasurements = demoArtifactSources.measurements as MeasurementsData;
export const demoValidationReport = demoArtifactSources.validation_report as ValidationReportData;

export const demoStatisticalModelSpec =
  demoArtifactSources.statistical_model_spec as unknown as StatisticalModelSpecData;
export const demoPosterior = demoArtifactSources.posterior as PosteriorData;
export const demoBaselineReport = demoArtifactSources.baseline_report as BaselineReportData;
