import type { ArtifactViewId, LLMTrace } from "@nof1-causal-lab/api-types";
import rawDataTrace from "../../../../../data/DEMO/run/temporal-llm/seq-000001/raw-data/trace.json";
import latentStructureTrace from "../../../../../data/DEMO/run/temporal-llm/seq-000003/latent-structure/trace.json";
import measurementStructureTrace from "../../../../../data/DEMO/run/temporal-llm/seq-000004/measurement-structure/trace.json";
import measurementsTrace from "../../../../../data/DEMO/run/temporal-llm/seq-000005/measurement-chunk-000000-attempt-001/trace.json";
import statisticalModelSpecTrace from "../../../../../data/DEMO/run/temporal-llm/seq-000006/statistical_model_spec/trace.json";
import baselineReportTrace from "../../../../../data/DEMO/run/temporal-llm/seq-000007/baseline-report/trace.json";

export const demoTraceRefs = {
  raw_data: "file:DEMO/run/temporal-llm/seq-000001/raw-data/trace.json",
  latent_structure: "file:DEMO/run/temporal-llm/seq-000003/latent-structure/trace.json",
  measurement_structure: "file:DEMO/run/temporal-llm/seq-000004/measurement-structure/trace.json",
  measurements:
    "file:DEMO/run/temporal-llm/seq-000005/measurement-chunk-000000-attempt-001/trace.json",
  statistical_model_spec: "file:DEMO/run/temporal-llm/seq-000006/statistical_model_spec/trace.json",
  baseline_report: "file:DEMO/run/temporal-llm/seq-000007/baseline-report/trace.json",
} as const satisfies Partial<Record<ArtifactViewId, string>>;

export const demoTraces = {
  raw_data: rawDataTrace as LLMTrace,
  latent_structure: latentStructureTrace as LLMTrace,
  measurement_structure: measurementStructureTrace as LLMTrace,
  measurements: measurementsTrace as LLMTrace,
  statistical_model_spec: statisticalModelSpecTrace as LLMTrace,
  baseline_report: baselineReportTrace as LLMTrace,
} as const satisfies Partial<Record<ArtifactViewId, LLMTrace>>;
