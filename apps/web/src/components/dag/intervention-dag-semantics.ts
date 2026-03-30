import type { ActionReferenceKind, Stage6SimulationResult } from "./intervention-dag-types";

function formatSignedAmount(value: number, digits = 1): string {
  const magnitude = Math.abs(value).toFixed(digits);
  return value >= 0 ? `+${magnitude}` : `-${magnitude}`;
}

function formatEvidenceTimestamp(value: string): string {
  return value.replace(/T.*$/, "");
}

export function getActionReference(result: Stage6SimulationResult): ActionReferenceKind {
  return result.rung === 3 ? "abducted_state" : "baseline_steady_state";
}

export function formatActionDescription(result: Stage6SimulationResult): string {
  const action = result.action;
  const treatment = action.variable;
  if (action.mode === "set") {
    return `do(${treatment} = ${Number(action.value ?? 0).toFixed(1)})`;
  }

  const reference = getActionReference(result) === "baseline_steady_state" ? "baseline" : "abducted state";
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
    : "from abducted state";
}

export function formatEvidenceWindowLabel(result: Stage6SimulationResult): string | null {
  if (result.rung !== 3) {
    return null;
  }
  const { start_time, end_time, n_timepoints } = result.evidence;
  return `conditioned on observed window ${formatEvidenceTimestamp(start_time)} to ${formatEvidenceTimestamp(end_time)} (${n_timepoints} points)`;
}

export function getEffectTrajectoryDays(result: Stage6SimulationResult): number[] {
  return result.effect_trajectory?.map((point) => point.day) ?? [];
}

export function getNodeEffectSeries(
  result: Stage6SimulationResult,
  nodeName: string,
): number[] | null {
  return result.visualization?.node_effect_trajectories?.[nodeName] ?? null;
}
