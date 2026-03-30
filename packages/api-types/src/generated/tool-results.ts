/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python tool result models via:
 *   cd apps/data-pipeline && uv run python scripts/export_schemas.py
 *   cd packages/api-types && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/causal_ssm_agent/flows/stages/contracts.py
 */

/**
 * Combined JSON Schema for declared tool result contracts.
 */
export type CausalSSMToolResults =
  | ToolErrorContract
  | EffectSummaryContract
  | EffectTrajectoryPointContract
  | Stage6VisualizationContract
  | CounterfactualEvidenceResultContract
  | SimulateInterventionResultContract
  | SimulateCounterfactualResultContract
  | SimulateInterventionToolResultContract
  | SimulateCounterfactualToolResultContract;
export type SimulateInterventionToolResultContract = SimulateInterventionResultContract | ToolErrorContract;
export type SimulateCounterfactualToolResultContract = SimulateCounterfactualResultContract | ToolErrorContract;

export interface ToolErrorContract {
  error: string;
  identifiable_treatments?: string[] | null;
}
export interface EffectSummaryContract {
  mean: number;
  median: number;
  lower_95: number;
  upper_95: number;
  prob_positive: number;
}
export interface EffectTrajectoryPointContract {
  day: number;
  effect: number;
}
export interface Stage6VisualizationContract {
  /**
   * Per-construct latent trajectories for the reference path aligned to effect_trajectory days. This is the no-action baseline forecast for rung-2 queries and the factual forecast from the abducted state for rung-3 queries.
   */
  reference_node_trajectories?: {
    [k: string]: number[] | undefined;
  } | null;
  /**
   * Per-construct latent trajectories under the queried action aligned to effect_trajectory days.
   */
  action_node_trajectories?: {
    [k: string]: number[] | undefined;
  } | null;
  /**
   * Per-construct latent effect trajectories aligned to effect_trajectory days. Values are causal deltas relative to the relevant reference path.
   */
  node_effect_trajectories?: {
    [k: string]: number[] | undefined;
  } | null;
  /**
   * Recovered latent state at the evidence boundary for rung-3 queries.
   */
  abducted_state?: {
    [k: string]: number | undefined;
  } | null;
}
export interface CounterfactualEvidenceResultContract {
  start_time: string;
  end_time: string;
  n_timepoints: number;
  variables: string[];
  conditioning_method: string;
}
export interface SimulateInterventionResultContract {
  action: InterventionActionInput;
  outcome: string;
  summary: EffectSummaryContract;
  effect_trajectory?: EffectTrajectoryPointContract[] | null;
  visualization?: Stage6VisualizationContract | null;
  manifest_effects?: {
    [k: string]: number | undefined;
  } | null;
  warnings: string[];
  rung: 2;
  estimand: "steady_state" | "trajectory";
  baseline_treatment_mean: number;
}
export interface InterventionActionInput {
  /**
   * Latent construct to intervene on.
   */
  variable: string;
  /**
   * 'set' clamps the construct to a value; 'shift' adds an amount to baseline.
   */
  mode: "set" | "shift";
  /**
   * Required when mode='set'. Absolute latent-space value to clamp to.
   */
  value?: number | null;
  /**
   * Required when mode='shift'. Additive latent-space delta from baseline.
   */
  amount?: number | null;
}
export interface SimulateCounterfactualResultContract {
  action: InterventionActionInput;
  outcome: string;
  summary: EffectSummaryContract;
  effect_trajectory?: EffectTrajectoryPointContract[] | null;
  visualization?: Stage6VisualizationContract | null;
  manifest_effects?: {
    [k: string]: number | undefined;
  } | null;
  warnings: string[];
  rung: 3;
  evidence: CounterfactualEvidenceResultContract;
  estimand: "end_state" | "trajectory";
  baseline_forecast_mean: number;
}
