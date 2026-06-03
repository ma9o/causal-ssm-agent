import type { Construct } from "@nof1-causal-lab/api-types";
import type { DagAnimationState } from "@/lib/hooks/use-dag-animation";
import {
  formatClampReferenceLabel,
  formatClampShortLabel,
  formatScenarioActionDescription,
  formatScenarioStartLabel,
  getClampByVariable,
  getEffectTrajectoryDays,
  getNodeActionSeries,
  getNodeEffectSeries,
  getNodeReferenceSeries,
  isAbductedStart,
} from "./intervention-dag-semantics";
import type { NodeAnimPhase, Stage6SimulationResult } from "./intervention-dag-types";

export type DagMode = "static" | "rung2" | "rung3";

/** Which trajectory the node sparklines emphasise (a presentational re-slice). */
export type DagMetric = "effect" | "action" | "reference";

/**
 * A baseline (do(treatment += 1 SD)) scenario rendered statically on the DAG —
 * treatment clamped, outcome highlighted with a scalar effect, no animation.
 */
export interface StaticScenarioInput {
  treatment: string;
  outcome: string;
  effectMagnitude: number;
  actionLabelShort: string;
  actionReferenceLabel: string;
}

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
  return isAbductedStart(result) ? "rung3" : "rung2";
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

function zeros(length: number): number[] {
  return Array.from({ length }, () => 0);
}

/**
 * Pick the node's emphasised (main) series and its dimmed baseline for a given
 * metric. `effect`/`reference` plot against a flat zero baseline; `action` plots
 * the action path against the reference path.
 */
function seriesForMetric(
  result: Stage6SimulationResult,
  nodeName: string,
  metric: DagMetric,
): { comparison: number[] | null; reference: number[] | null } {
  if (metric === "effect") {
    const effect = getNodeEffectSeries(result, nodeName);
    return effect
      ? { comparison: effect, reference: zeros(effect.length) }
      : { comparison: null, reference: null };
  }
  if (metric === "reference") {
    const reference = getNodeReferenceSeries(result, nodeName);
    return reference
      ? { comparison: reference, reference: zeros(reference.length) }
      : { comparison: null, reference: null };
  }
  return {
    comparison: getNodeActionSeries(result, nodeName),
    reference: getNodeReferenceSeries(result, nodeName),
  };
}

function emptyNodeViewModel(): EffectNodeViewModel {
  return {
    animPhase: "idle",
    effectMagnitude: null,
    startStateValue: null,
    timeIndex: 0,
    timeStepsDays: null,
    referenceTimeSeries: null,
    comparisonTimeSeries: null,
    actionLabelShort: null,
    actionReferenceLabel: null,
  };
}

function buildStaticViewModel(
  constructs: Construct[],
  scenario: StaticScenarioInput,
): InterventionDagViewModel {
  return {
    mode: "static",
    actionDescription: null,
    startDescription: null,
    timeStepsDays: [],
    nodeData: Object.fromEntries(
      constructs.map((construct) => {
        if (construct.name === scenario.treatment) {
          return [
            construct.name,
            {
              ...emptyNodeViewModel(),
              rung: 2,
              animPhase: "clamped",
              actionLabelShort: scenario.actionLabelShort,
              actionReferenceLabel: scenario.actionReferenceLabel,
            },
          ];
        }
        if (construct.name === scenario.outcome) {
          return [
            construct.name,
            {
              ...emptyNodeViewModel(),
              rung: 2,
              animPhase: "active",
              effectMagnitude: scenario.effectMagnitude,
            },
          ];
        }
        return [construct.name, emptyNodeViewModel()];
      }),
    ),
  };
}

export function buildInterventionDagViewModel(args: {
  constructs: Construct[];
  requestedHorizonDays?: number;
  result: Stage6SimulationResult | null;
  staticScenario?: StaticScenarioInput | null;
  metric?: DagMetric;
  animation: Pick<
    DagAnimationState,
    "phase" | "timeIndex" | "nodePhases" | "nodeEffects" | "startStateValues"
  >;
}): InterventionDagViewModel {
  const {
    constructs,
    requestedHorizonDays,
    result,
    staticScenario,
    metric = "effect",
    animation,
  } = args;

  if (!result) {
    if (staticScenario) {
      return buildStaticViewModel(constructs, staticScenario);
    }
    return {
      mode: "static",
      actionDescription: null,
      startDescription: null,
      timeStepsDays: [],
      nodeData: {},
    };
  }

  const mode = getDagMode(result);
  const timeStepsDays = getEffectTrajectoryDays(result);
  const clampByVariable = getClampByVariable(result);
  const rung: 2 | 3 = isAbductedStart(result) ? 3 : 2;

  return {
    mode,
    actionDescription: formatScenarioActionDescription(result),
    startDescription: formatScenarioStartLabel(result),
    timeStepsDays,
    temporalMarkers: buildTemporalMarkers(timeStepsDays, requestedHorizonDays),
    nodeData: Object.fromEntries(
      constructs.map((construct) => {
        const { comparison, reference } = seriesForMetric(result, construct.name, metric);
        const clamp = clampByVariable.get(construct.name);
        return [
          construct.name,
          {
            rung,
            animPhase: animation.nodePhases[construct.name] ?? "idle",
            effectMagnitude: animation.nodeEffects[construct.name] ?? null,
            startStateValue: animation.startStateValues[construct.name] ?? null,
            timeIndex: animation.timeIndex,
            timeStepsDays,
            referenceTimeSeries: reference,
            comparisonTimeSeries: comparison,
            actionLabelShort: clamp ? formatClampShortLabel(clamp) : null,
            actionReferenceLabel: clamp ? formatClampReferenceLabel(result, clamp) : null,
          },
        ];
      }),
    ),
  };
}
