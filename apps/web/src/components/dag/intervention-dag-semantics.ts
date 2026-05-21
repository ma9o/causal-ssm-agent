import type { ActionReferenceKind, Stage6SimulationResult } from "./intervention-dag-types";

function formatSignedAmount(value: number, digits = 1): string {
  const magnitude = Math.abs(value).toFixed(digits);
  return value >= 0 ? `+${magnitude}` : `-${magnitude}`;
}

function formatStartTimestamp(value: string): string {
  return value.replace(/T.*$/, "");
}

export function getActionReference(result: Stage6SimulationResult): ActionReferenceKind {
  return result.rung === 3 ? "fitted_start_state" : "baseline_steady_state";
}

export function formatActionDescription(result: Stage6SimulationResult): string {
  const action = result.action;
  const treatment = action.variable;
  if (action.mode === "set") {
    return `do(${treatment} = ${Number(action.value ?? 0).toFixed(1)})`;
  }

  const reference =
    getActionReference(result) === "baseline_steady_state" ? "baseline" : "fitted start state";
  return `do(${treatment} = ${reference} ${formatSignedAmount(action.amount ?? 0)})`;
}

export function formatActionShortLabel(action: Stage6SimulationResult["action"]): string {
  if (action.mode === "set") {
    return `set ${Number(action.value ?? 0).toFixed(1)}`;
  }
  return `shift ${formatSignedAmount(action.amount ?? 0)}`;
}

export function formatActionReferenceLabel(result: Stage6SimulationResult): string {
  if (result.action.mode === "set") {
    return "absolute latent value";
  }
  return getActionReference(result) === "baseline_steady_state"
    ? "from baseline steady state"
    : "from fitted start state";
}

export function formatCounterfactualStartLabel(result: Stage6SimulationResult): string | null {
  if (result.rung !== 3) {
    return null;
  }
  const { time, time_index } = result.start;
  if (time) {
    return `started from fitted state ${formatStartTimestamp(time)} (#${time_index})`;
  }
  return `started from fitted state #${time_index}`;
}

export function getEffectTrajectoryDays(result: Stage6SimulationResult): number[] {
  return result.effect_trajectory?.map((point) => point.day) ?? [];
}

export function getNodeReferenceSeries(
  result: Stage6SimulationResult,
  nodeName: string,
): number[] | null {
  return result.visualization?.reference_node_trajectories?.[nodeName] ?? null;
}

export function getNodeActionSeries(
  result: Stage6SimulationResult,
  nodeName: string,
): number[] | null {
  return result.visualization?.action_node_trajectories?.[nodeName] ?? null;
}

export function getNodeEffectSeries(
  result: Stage6SimulationResult,
  nodeName: string,
): number[] | null {
  return result.visualization?.node_effect_trajectories?.[nodeName] ?? null;
}
