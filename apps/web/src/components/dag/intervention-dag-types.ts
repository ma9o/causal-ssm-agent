import type { LatentClampInput, SimulateScenarioResult } from "@nof1-causal-lab/api-types";

export interface EdgePosterior {
  mean: number;
  ci_lower: number;
  ci_upper: number;
}

/** A composable Stage 6 scenario result: a start state + a list of timed latent clamps. */
export type Stage6SimulationResult = SimulateScenarioResult;

/** One do-operator clamp within a scenario. */
export type LatentClamp = LatentClampInput;
