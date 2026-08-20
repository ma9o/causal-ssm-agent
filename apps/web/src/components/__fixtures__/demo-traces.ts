import type { ArtifactViewId, LLMTrace } from "@nof1-causal-lab/api-types";
import baselineReportTrace from "../../../../../data/DEMO/fixture/traces/baseline_report.json";
import latentStructureTrace from "../../../../../data/DEMO/fixture/traces/latent_structure.json";
import measurementStructureTrace from "../../../../../data/DEMO/fixture/traces/measurement_structure.json";
import measurementsTrace from "../../../../../data/DEMO/fixture/traces/measurements.json";
import rawDataTrace from "../../../../../data/DEMO/fixture/traces/raw_data.json";
import statisticalModelSpecTrace from "../../../../../data/DEMO/fixture/traces/statistical_model_spec.json";

export const demoTraces = {
  raw_data: rawDataTrace as LLMTrace,
  latent_structure: latentStructureTrace as LLMTrace,
  measurement_structure: measurementStructureTrace as LLMTrace,
  measurements: measurementsTrace as LLMTrace,
  statistical_model_spec: statisticalModelSpecTrace as LLMTrace,
  baseline_report: baselineReportTrace as LLMTrace,
} as const satisfies Partial<Record<ArtifactViewId, LLMTrace>>;
