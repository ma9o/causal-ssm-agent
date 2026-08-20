import type { BaselineReportData, LLMTrace } from "@nof1-causal-lab/api-types";
import {
  demoBaselineReport,
  demoLatentStructure,
  demoPosterior,
  demoStatisticalModelSpec,
} from "../../__fixtures__/demo-artifacts";
import { demoTraces } from "../../__fixtures__/demo-traces";
import { deriveConstructStatuses } from "../construct-statuses";
import {
  buildEdgePosteriors,
  buildPersistencePosteriors,
} from "../../pipeline/output-views/baseline-report-scenarios";
import {
  constructs,
  design,
  edges,
  indicators,
  knownInputs,
  structuralPlan,
} from "./dag-base-fixtures";

export { constructs, edges, indicators, knownInputs };

export const edgePosteriors = buildEdgePosteriors({
  latentStructure: demoLatentStructure,
  modelSpec: demoStatisticalModelSpec,
  posterior: demoPosterior,
});
export const persistencePosteriors = buildPersistencePosteriors({
  modelSpec: demoStatisticalModelSpec,
  posterior: demoPosterior,
});

const demo = demoBaselineReport as BaselineReportData;
export const identifiableTreatments = demo.intervention_results.map(({ treatment }) => treatment);
export const nodeStatuses = deriveConstructStatuses(design, structuralPlan);

export const demoBaselineTrace: LLMTrace = demoTraces.baseline_report;

/** The complete materialized analysis artifact (rankings, scenarios, and summary). */
export const materializedBaselineReportData: BaselineReportData = {
  intervention_results: demo.intervention_results,
  saved_scenarios: demo.saved_scenarios ?? null,
  final_summary: demo.final_summary ?? null,
};
