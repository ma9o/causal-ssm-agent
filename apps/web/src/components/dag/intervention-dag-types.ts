import type {
  SimulateCounterfactualResult,
  SimulateInterventionResult,
} from "@causal-ssm/api-types";

export interface EdgePosterior {
  mean: number;
  ci_lower: number;
  ci_upper: number;
}

export type EdgeAnimState = "normal" | "cut" | "flowing" | "dimmed";

export type NodeAnimPhase = "idle" | "clamped" | "receiving" | "active" | "abducted" | "dimmed";

export type ActionReferenceKind = "baseline_steady_state" | "abducted_state";

export type Stage6SimulationResult =
  | SimulateInterventionResult
  | SimulateCounterfactualResult;
