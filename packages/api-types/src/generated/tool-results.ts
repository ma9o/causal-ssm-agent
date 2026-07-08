/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python tool result models via:
 *   cd apps/data-pipeline && uv run python scripts/export_schemas.py
 *   cd packages/api-types && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/nof1_causal_lab/flows/artifact_contracts.py
 */

/**
 * Combined JSON Schema for declared tool result contracts.
 */
export type CausalSSMToolResults =
  | ToolErrorContract
  | EffectSummaryContract
  | EffectTrajectoryPointContract
  | BaselineReportVisualizationContract
  | ScenarioStartResultContract
  | SimulateScenarioResultContract
  | SimulateScenarioToolResultContract;
export type SimulateScenarioToolResultContract = SimulateScenarioResultContract | ToolErrorContract;

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
export interface BaselineReportVisualizationContract {
  /**
   * Per-construct latent trajectories for the reference (no-clamp) path aligned to effect_trajectory days.
   */
  reference_node_trajectories?: {
    [k: string]: number[] | undefined;
  } | null;
  /**
   * Per-construct latent trajectories under the composed clamps aligned to effect_trajectory days.
   */
  action_node_trajectories?: {
    [k: string]: number[] | undefined;
  } | null;
  /**
   * Per-construct latent effect trajectories aligned to effect_trajectory days. Values are causal deltas relative to the reference path.
   */
  node_effect_trajectories?: {
    [k: string]: number[] | undefined;
  } | null;
  /**
   * Posterior mean latent state the rollout started from.
   */
  start_state?: {
    [k: string]: number | undefined;
  } | null;
}
export interface ScenarioStartResultContract {
  kind: "baseline" | "abducted";
  time_index?: number | null;
  time?: string | null;
  state_source: "baseline_steady_state" | "fitted_latent_paths";
}
export interface SimulateScenarioResultContract {
  start: ScenarioStartResultContract;
  clamps: LatentClampInput[];
  outcome: string;
  estimand: "end_state" | "trajectory";
  summary: EffectSummaryContract;
  effect_trajectory?: EffectTrajectoryPointContract[] | null;
  visualization?: BaselineReportVisualizationContract | null;
  manifest_effects?: {
    [k: string]: number | undefined;
  } | null;
  /**
   * Mean reference outcome (baseline steady state or factual forecast).
   */
  reference_mean: number;
  warnings: string[];
}
/**
 * A do-operator on one latent variable over a time window.
 *
 * The window is ``[from_day, to_day)`` in days relative to the rollout start; outside
 * the window the variable evolves under its natural dynamics. ``set`` pins to an absolute
 * value, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly
 * interpolates across the window, and ``trajectory`` tracks a list of values across it.
 */
export interface LatentClampInput {
  /**
   * Latent construct to clamp.
   */
  variable: string;
  /**
   * How the clamped value is specified over the window.
   */
  mode: "set" | "shift" | "ramp" | "trajectory";
  /**
   * Required when mode='set'. Absolute latent-space value.
   */
  value?: number | null;
  /**
   * Required when mode='shift'. Additive delta from the start-state value.
   */
  amount?: number | null;
  /**
   * Required when mode='ramp'. Value at from_day.
   */
  value_start?: number | null;
  /**
   * Required when mode='ramp'. Value at to_day.
   */
  value_end?: number | null;
  /**
   * Required when mode='trajectory'. Values sampled evenly across the window.
   */
  values?: number[] | null;
  /**
   * Window onset in days from the rollout start.
   */
  from_day: number;
  /**
   * Window end in days from the rollout start. Null runs through the horizon.
   */
  to_day?: number | null;
}
