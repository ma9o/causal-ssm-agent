import type {
  ActionReferenceKind,
  LatentClamp,
  Stage6SimulationResult,
} from "./intervention-dag-types";

function formatSignedAmount(value: number, digits = 1): string {
  const magnitude = Math.abs(value).toFixed(digits);
  return value >= 0 ? `+${magnitude}` : `-${magnitude}`;
}

function formatStartTimestamp(value: string): string {
  return value.replace(/T.*$/, "");
}

/** True when the rollout starts from an abducted individual state (counterfactual). */
export function isAbductedStart(result: Stage6SimulationResult): boolean {
  return result.start.kind === "abducted";
}

export function getActionReference(result: Stage6SimulationResult): ActionReferenceKind {
  return isAbductedStart(result) ? "fitted_start_state" : "baseline_steady_state";
}

/** Clamps keyed by the latent variable they act on. */
export function getClampByVariable(result: Stage6SimulationResult): Map<string, LatentClamp> {
  return new Map(result.clamps.map((clamp) => [clamp.variable, clamp]));
}

export function getClampedVariables(result: Stage6SimulationResult): string[] {
  return result.clamps.map((clamp) => clamp.variable);
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

/** Window descriptor, or null for a full-horizon clamp opening at the start. */
export function formatClampWindow(clamp: LatentClamp): string | null {
  const from = clamp.from_day ?? 0;
  const to = clamp.to_day;
  if (from === 0 && to == null) {
    return null;
  }
  return `d${from}–${to == null ? "∞" : to}`;
}

export function formatClampShortLabel(clamp: LatentClamp): string {
  return formatClampValue(clamp);
}

export function formatClampReferenceLabel(
  result: Stage6SimulationResult,
  clamp: LatentClamp,
): string {
  const base =
    clamp.mode === "set"
      ? "absolute latent value"
      : getActionReference(result) === "baseline_steady_state"
        ? "from baseline"
        : "from fitted start state";
  const window = formatClampWindow(clamp);
  return window ? `${base} · ${window}` : base;
}

/** do(...) description joining every clamp in the scenario. */
export function formatScenarioActionDescription(result: Stage6SimulationResult): string {
  return result.clamps
    .map((clamp) => `do(${clamp.variable} ${formatClampValue(clamp)})`)
    .join(", ");
}

export function formatScenarioStartLabel(result: Stage6SimulationResult): string | null {
  if (!isAbductedStart(result)) {
    return null;
  }
  const { time, time_index } = result.start;
  if (time) {
    return `from fitted state ${formatStartTimestamp(time)} (#${time_index})`;
  }
  return time_index != null ? `from fitted state #${time_index}` : "from fitted state";
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
