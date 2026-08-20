import type { LatentClamp, AnalysisSimulationResult } from "./intervention-dag-types";

function formatSignedAmount(value: number, digits = 1): string {
  const magnitude = Math.abs(value).toFixed(digits);
  return value >= 0 ? `+${magnitude}` : `-${magnitude}`;
}

function requiredClampNumber(
  clamp: LatentClamp,
  field: "value" | "amount" | "value_start" | "value_end",
): number {
  const value = clamp[field];
  if (value == null) {
    throw new Error(`Invalid ${clamp.mode} clamp from backend: ${field} is required`);
  }
  return value;
}

function requiredClampTrajectory(clamp: LatentClamp): number[] {
  if (clamp.values == null || clamp.values.length < 2) {
    throw new Error("Invalid trajectory clamp from backend: values requires at least two points");
  }
  return clamp.values;
}

/** Compact value descriptor for a single clamp (e.g. `shift +1.0`, `set 0.5`). */
export function formatClampValue(clamp: LatentClamp): string {
  switch (clamp.mode) {
    case "set":
      return `set ${requiredClampNumber(clamp, "value").toFixed(1)}`;
    case "shift":
      return `shift ${formatSignedAmount(requiredClampNumber(clamp, "amount"))}`;
    case "ramp":
      return `ramp ${requiredClampNumber(clamp, "value_start").toFixed(1)}→${requiredClampNumber(clamp, "value_end").toFixed(1)}`;
    default:
      return `trajectory (${requiredClampTrajectory(clamp).length} pts)`;
  }
}

/** do(...) description joining every clamp in the scenario. */
export function formatScenarioActionDescription(result: AnalysisSimulationResult): string {
  return result.clamps
    .map((clamp) => `do(${clamp.variable} ${formatClampValue(clamp)})`)
    .join(", ");
}

export function getEffectTrajectoryDays(result: AnalysisSimulationResult): number[] {
  return result.effect_trajectory?.map((point) => point.day) ?? [];
}

export function getNodeReferenceSeries(
  result: AnalysisSimulationResult,
  nodeName: string,
): number[] | null {
  return result.visualization?.reference_node_trajectories?.[nodeName] ?? null;
}

export function getNodeActionSeries(
  result: AnalysisSimulationResult,
  nodeName: string,
): number[] | null {
  return result.visualization?.action_node_trajectories?.[nodeName] ?? null;
}
