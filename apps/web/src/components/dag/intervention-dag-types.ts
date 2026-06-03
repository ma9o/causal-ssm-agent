import type { LatentClampInput, SimulateScenarioResult } from "@nof1-causal-lab/api-types";

export interface EdgePosterior {
  mean: number;
  ci_lower: number;
  ci_upper: number;
}

export type EdgeAnimState = "normal" | "cut" | "flowing" | "dimmed";

export type NodeAnimPhase = "idle" | "clamped" | "receiving" | "active" | "start_state" | "dimmed";

export type ActionReferenceKind = "baseline_steady_state" | "fitted_start_state";

/** A composable Stage 6 scenario result: a start state + a list of timed latent clamps. */
export type Stage6SimulationResult = SimulateScenarioResult;

/** One do-operator clamp within a scenario. */
export type LatentClamp = LatentClampInput;
