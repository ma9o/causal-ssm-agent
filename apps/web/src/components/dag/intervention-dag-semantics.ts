import type { LatentClamp, AnalysisSimulationResult } from "./intervention-dag-types";

function formatSignedAmount(value: number, digits = 1): string {
  const magnitude = Math.abs(value).toFixed(digits);
  return value >= 0 ? `+${magnitude}` : `-${magnitude}`;
}

/** Compact value descriptor for a single clamp (e.g. `shift +1.0`, `set 0.5`). */
export function formatClampValue(clamp: LatentClamp): string {
  switch (clamp.mode) {
    case "set":
      return `set ${Number(clamp.value ?? 0).toFixed(1)}`;
    case "shift":
      return `shift ${formatSignedAmount(clamp.amount ?? 0)}`;
    case "ramp":
      return `ramp ${Number(clamp.value_start ?? 0).toFixed(1)}→${Number(clamp.value_end ?? 0).toFixed(1)}`;
    default:
      return `trajectory (${clamp.values?.length ?? 0} pts)`;
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
