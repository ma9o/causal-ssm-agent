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

const { store, temporaryDemoRun } = demoArtifactSources;

export const demoRawData = store.raw_data as RawDataData;
export const demoLatentStructure = store.latent_structure as LatentStructureData;
export const demoMeasurementStructure = {
  ...store.measurement_structure,
  causal_design: store.causal_design,
} as unknown as MeasurementStructureViewData;
export const demoMeasurements = store.measurements as MeasurementsData;
export const demoValidationReport = store.validation_report as ValidationReportData;

export const demoStatisticalModelSpec =
  temporaryDemoRun.statistical_model_spec as unknown as StatisticalModelSpecData;
export const demoPosterior = temporaryDemoRun.posterior as PosteriorData;
export const demoBaselineReport = temporaryDemoRun.baseline_report as BaselineReportData;
