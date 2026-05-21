import type { Construct } from "@nof1-causal-lab/api-types";
import type { DagAnimationState } from "@/lib/hooks/use-dag-animation";
import {
  formatActionDescription,
  formatActionReferenceLabel,
  formatActionShortLabel,
  formatCounterfactualStartLabel,
  getEffectTrajectoryDays,
  getNodeActionSeries,
  getNodeReferenceSeries,
} from "./intervention-dag-semantics";
import type { NodeAnimPhase, Stage6SimulationResult } from "./intervention-dag-types";

export type DagMode = "static" | "rung2" | "rung3";

export interface TemporalMarker {
  day: number;
  label: string;
}

export interface EffectNodeViewModel {
  rung?: 2 | 3;
  animPhase: NodeAnimPhase;
  effectMagnitude: number | null;
  startStateValue: number | null;
  timeIndex: number;
  timeStepsDays: number[] | null;
  referenceTimeSeries: number[] | null;
  comparisonTimeSeries: number[] | null;
  actionLabelShort: string | null;
  actionReferenceLabel: string | null;
}

export interface InterventionDagViewModel {
  mode: DagMode;
  actionDescription: string | null;
  startDescription: string | null;
  timeStepsDays: number[];
  temporalMarkers?: TemporalMarker[];
  nodeData: Record<string, EffectNodeViewModel>;
}

function getDagMode(result: Stage6SimulationResult | null): DagMode {
  if (!result) {
    return "static";
  }
  return result.rung === 3 ? "rung3" : "rung2";
}

function formatDayLabel(day: number): string {
  const rounded = Math.round(day * 10) / 10;
  return Number.isInteger(rounded) ? `${rounded}d` : `${rounded.toFixed(1)}d`;
}

function hasDay(targetDay: number, timeStepsDays: number[]): boolean {
  return timeStepsDays.some((day) => Math.abs(day - targetDay) < 1e-6);
}

function buildTemporalMarkers(
  timeStepsDays: number[],
  requestedHorizonDays?: number,
): TemporalMarker[] | undefined {
  if (timeStepsDays.length === 0) {
    return undefined;
  }
  const startDay = timeStepsDays[0] ?? 0;
  const endDay = timeStepsDays[timeStepsDays.length - 1] ?? startDay;
  const requestedEndDay =
    requestedHorizonDays != null && requestedHorizonDays > 0 ? requestedHorizonDays : endDay;
  const standardMilestones = [1, 7, 30, 90, 180, 365].filter(
    (day) => day > startDay && day < requestedEndDay && hasDay(day, timeStepsDays),
  );

  return Array.from(
    new Map(
      [startDay, ...standardMilestones, endDay]
        .filter((day, index, days) => {
          if (index === 0) {
            return true;
          }
          return !days.slice(0, index).some((seen) => Math.abs(seen - day) < 1e-6);
        })
        .sort((left, right) => left - right)
        .map((day) => [day.toFixed(3), { day, label: formatDayLabel(day) }]),
    ).values(),
  );
}

export function buildInterventionDagViewModel(args: {
  constructs: Construct[];
  requestedHorizonDays?: number;
  result: Stage6SimulationResult | null;
  animation: Pick<DagAnimationState, "phase" | "timeIndex" | "nodePhases" | "nodeEffects" | "startStateValues">;
}): InterventionDagViewModel {
  const { constructs, requestedHorizonDays, result, animation } = args;
  const mode = getDagMode(result);
  if (!result) {
    return {
      mode,
      actionDescription: null,
      startDescription: null,
      timeStepsDays: [],
      nodeData: {},
    };
  }

  const timeStepsDays = getEffectTrajectoryDays(result);
  const actionLabelShort = formatActionShortLabel(result.action);
  const actionReferenceLabel = formatActionReferenceLabel(result);

  return {
    mode,
    actionDescription: formatActionDescription(result),
    startDescription: formatCounterfactualStartLabel(result),
    timeStepsDays,
    temporalMarkers: buildTemporalMarkers(timeStepsDays, requestedHorizonDays),
    nodeData: Object.fromEntries(
      constructs.map((construct) => [
        construct.name,
        {
          rung: result.rung,
          animPhase: animation.nodePhases[construct.name] ?? "idle",
          effectMagnitude: animation.nodeEffects[construct.name] ?? null,
          startStateValue: animation.startStateValues[construct.name] ?? null,
          timeIndex: animation.timeIndex,
          timeStepsDays,
          referenceTimeSeries: getNodeReferenceSeries(result, construct.name),
          comparisonTimeSeries: getNodeActionSeries(result, construct.name),
          actionLabelShort,
          actionReferenceLabel,
        },
      ]),
    ),
  };
}
