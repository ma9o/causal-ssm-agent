import type { Stage6SimulationResult } from "../intervention-dag-types";

/**
 * PROPOSED extension to `Stage6VisualizationContract` — the fields the
 * interactive DAG's rich layers need, which the current contract does NOT carry.
 * This is the concrete spec for the data-pipeline worktree to implement
 * (computed by the exact SSM engines, never client-side). Everything is optional:
 * the renderers hide a layer when its field is absent, so prod ships nothing fake.
 */

/** Per-edge drift decomposition (drives the drift glyph + edge color/width). */
export interface EdgeDrift {
  cause: string;
  effect: string;
  form: "linear" | "hill" | "mult";
  /** c(s) sampled over s∈[0,1] — panel A of the glyph (transfer shape). */
  transfer: number[];
  /** Signed drift contribution over the effect-trajectory days — panel B. */
  contribution: number[];
  /** Driver (cause) level over days — panel-A operating point's x. */
  driver_level: number[];
}

/** Per-construct self-effect (NodePotential −dV/dη) on the t−1 self-edge. */
export interface SelfEffect {
  node: string;
  /** −dV/dη sampled over level s∈[0,1] — panel A. */
  transfer: number[];
  /** Self-drift over days — panel B. */
  contribution: number[];
  /** Own level over days. */
  level: number[];
}

/** Per-indicator measurement channel (the manifest layer). */
export interface IndicatorSeries {
  construct: string;
  id: string;
  type: "cont" | "binary";
  sd: number;
  /** Observed & fit data points (revealed up to the playhead). */
  observed: { t: number; v: number }[];
  /** Factual model fit (the baseline the points scatter around). */
  ref: number[];
  /** Counterfactual posterior-predictive mean under do(). */
  cf: number[];
}

type Visualization = NonNullable<Stage6SimulationResult["visualization"]>;

export interface Stage6VisualizationExt extends Visualization {
  /** Per-construct realized factual latent path (revealed up to the playhead). */
  node_realized?: { [k: string]: number[] | undefined } | null;
  edge_drift?: EdgeDrift[] | null;
  self_effects?: SelfEffect[] | null;
  indicators?: IndicatorSeries[] | null;
}

const ext = (result: Stage6SimulationResult): Stage6VisualizationExt | null =>
  (result.visualization ?? null) as Stage6VisualizationExt | null;

export function getNodeRealized(result: Stage6SimulationResult, node: string): number[] | null {
  return ext(result)?.node_realized?.[node] ?? null;
}

export function getAllEdgeDrift(result: Stage6SimulationResult): EdgeDrift[] {
  return ext(result)?.edge_drift ?? [];
}

export function getEdgeDrift(
  result: Stage6SimulationResult,
  cause: string,
  effect: string,
): EdgeDrift | null {
  return ext(result)?.edge_drift?.find((e) => e.cause === cause && e.effect === effect) ?? null;
}

export function getSelfEffect(result: Stage6SimulationResult, node: string): SelfEffect | null {
  return ext(result)?.self_effects?.find((s) => s.node === node) ?? null;
}

export function getAllSelfEffects(result: Stage6SimulationResult): SelfEffect[] {
  return ext(result)?.self_effects ?? [];
}

export function getNodeIndicators(result: Stage6SimulationResult, node: string): IndicatorSeries[] {
  return (ext(result)?.indicators ?? []).filter((ind) => ind.construct === node);
}
