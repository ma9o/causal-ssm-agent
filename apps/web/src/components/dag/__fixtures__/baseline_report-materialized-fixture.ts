import type { BaselineReportData, LLMTrace } from "@nof1-causal-lab/api-types";
import { demoBaselineReport } from "../../__fixtures__/demo-artifacts";
import { demoTraces } from "../../__fixtures__/demo-traces";
import { constructs, edges, indicators } from "./dag-base-fixtures";
import { edgePosteriors } from "./intervention-dag-fixture";

export { constructs, edgePosteriors, edges, indicators };

const demo = demoBaselineReport as BaselineReportData;

export const demoBaselineTrace: LLMTrace = demoTraces.baseline_report;

/** The complete materialized analysis artifact (baselines + sims + summary). */
export const materializedBaselineReportData: BaselineReportData = {
  llm_trace_ref: demo.llm_trace_ref ?? null,
  intervention_results: demo.intervention_results,
  saved_scenarios: demo.saved_scenarios ?? null,
  final_summary: demo.final_summary ?? null,
};

export const outcomeName = "affective_state";
