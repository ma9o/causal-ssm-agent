import type { LatentClampInput } from "@nof1-causal-lab/api-types";
import type { AnalysisSimulationResult } from "../intervention-dag-types";

/**
 * Input to the baseline_report `simulate` tool. There is no generated TS interface for
 * it (the contract ships it as a JSON schema only), so we mirror the shape here.
 */
export interface SimulateInput {
  start: { kind: "baseline" | "abducted"; time_index?: number | null; time?: string | null };
  clamps: [LatentClampInput, ...LatentClampInput[]];
  outcome: string;
  query: {
    estimand: "trajectory" | "end_state";
    horizon_days: number;
    projection: "latent" | "manifest" | "both";
  };
}

/**
 * Runs a scenario and returns its result. The interactive DAG is agnostic to
 * how: production injects a `POST /api/tools/dispatch` call (the non-LLM tool
 * seam). Absent this, the DAG is a read-only viewer.
 */
export type SimulateFn = (input: SimulateInput) => Promise<AnalysisSimulationResult>;

/** Re-run `base`'s scenario start with a new set of clamps over the same horizon. */
export function buildSimulateInput(
  base: AnalysisSimulationResult,
  clamps: [LatentClampInput, ...LatentClampInput[]],
  horizonDays: number,
): SimulateInput {
  let start: SimulateInput["start"] = { kind: "baseline" };
  if (base.start.kind === "abducted") {
    start =
      base.start.time_index != null
        ? { kind: "abducted", time_index: base.start.time_index }
        : base.start.time != null
          ? { kind: "abducted", time: base.start.time }
          : { kind: "abducted" };
  }
  return {
    start,
    clamps,
    outcome: base.outcome,
    query: { estimand: "trajectory", horizon_days: horizonDays, projection: "latent" },
  };
}
