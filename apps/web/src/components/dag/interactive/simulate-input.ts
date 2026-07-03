import type { LatentClampInput } from "@nof1-causal-lab/api-types";
import type { Stage6SimulationResult } from "../intervention-dag-types";

/**
 * Input to the stage-6 `simulate` tool. There is no generated TS interface for
 * it (the contract ships it as a JSON schema only), so we mirror the shape here.
 */
export interface SimulateInput {
  start: { kind: "baseline" | "abducted"; time_index?: number | null; time?: string | null };
  clamps: LatentClampInput[];
  outcome: string;
  query: {
    estimand: "trajectory" | "end_state";
    horizon_days: number;
    projection: "latent" | "manifest" | "both";
  };
}

/**
 * Runs a scenario and returns its result. The interactive DAG is agnostic to
 * how: Storybook injects a mock; prod injects a `POST /api/tools/dispatch`
 * call (the non-LLM tool seam). Absent this, the DAG is a read-only viewer.
 */
export type SimulateFn = (input: SimulateInput) => Promise<Stage6SimulationResult>;

/** Re-run `base`'s scenario start with a new set of clamps over the same horizon. */
export function buildSimulateInput(
  base: Stage6SimulationResult,
  clamps: LatentClampInput[],
  horizonDays: number,
): SimulateInput {
  const start =
    base.start.kind === "abducted"
      ? { kind: "abducted" as const, time_index: base.start.time_index, time: base.start.time }
      : { kind: "baseline" as const };
  return {
    start,
    clamps,
    outcome: base.outcome,
    query: { estimand: "trajectory", horizon_days: horizonDays, projection: "latent" },
  };
}
