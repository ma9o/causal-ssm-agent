import type { ArtifactViewId, LLMTrace } from "@nof1-causal-lab/api-types";
import rawDataTrace from "../../../../../data/DEMO/episode/traces/000001/raw-data.json";
import latentStructureTrace from "../../../../../data/DEMO/episode/traces/000003/latent-structure.json";
import measurementStructureTrace from "../../../../../data/DEMO/episode/traces/000004/measurement-structure.json";
import measurementsTrace from "../../../../../data/DEMO/episode/traces/000005/measurement-chunk-000000-attempt-001.json";
import statisticalModelSpecTrace from "../../../../../data/DEMO/episode/traces/000006/statistical_model_spec.json";
import baselineReportTrace from "./demo-run/baseline_report_trace.json";

export const demoTraces = {
  raw_data: rawDataTrace as LLMTrace,
  latent_structure: latentStructureTrace as LLMTrace,
  measurement_structure: measurementStructureTrace as LLMTrace,
  measurements: measurementsTrace as LLMTrace,
  statistical_model_spec: statisticalModelSpecTrace as LLMTrace,
  baseline_report: baselineReportTrace as LLMTrace,
} as const satisfies Partial<Record<ArtifactViewId, LLMTrace>>;
