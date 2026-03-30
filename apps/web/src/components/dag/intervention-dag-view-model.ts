import type { Construct } from "@causal-ssm/api-types";
import type { DagAnimationState } from "@/lib/hooks/use-dag-animation";
import {
  formatActionDescription,
  formatActionReferenceLabel,
  formatActionShortLabel,
  formatEvidenceWindowLabel,
  getEffectTrajectoryDays,
  getNodeEffectSeries,
} from "./intervention-dag-semantics";
import type { NodeAnimPhase, Stage6SimulationResult } from "./intervention-dag-types";

export type DagMode = "static" | "rung2" | "rung3";

export interface TemporalMarker {
  day: number;
  label: string;
}

export interface PhaseMarker {
  position: number;
  label: string;
  active: boolean;
}

export interface EffectNodeViewModel {
  rung?: 2 | 3;
  animPhase: NodeAnimPhase;
  effectMagnitude: number | null;
  abductedValue: number | null;
  timeIndex: number;
  timeStepsDays: number[] | null;
  effectTimeSeries: number[] | null;
  actionLabelShort: string | null;
  actionReferenceLabel: string | null;
}

export interface InterventionDagViewModel {
  mode: DagMode;
  actionDescription: string | null;
  evidenceDescription: string | null;
  timeStepsDays: number[];
  temporalMarkers?: TemporalMarker[];
  phaseMarkers?: PhaseMarker[];
  nodeData: Record<string, EffectNodeViewModel>;
}

function getDagMode(result: Stage6SimulationResult | null): DagMode {
  if (!result) {
    return "static";
  }
  return result.rung === 3 ? "rung3" : "rung2";
}

function buildTemporalMarkers(
  result: Stage6SimulationResult,
  timeStepsDays: number[],
): TemporalMarker[] | undefined {
  if (timeStepsDays.length === 0) {
    return undefined;
  }
  const max = timeStepsDays[timeStepsDays.length - 1] ?? 0;
  const markers: TemporalMarker[] = [
    { day: 1, label: "1d" },
    { day: 7, label: "7d" },
    { day: 30, label: "30d" },
  ];

  const peakPoint = result.effect_trajectory?.reduce<
    { day: number; magnitude: number } | null
  >((currentPeak, point) => {
    const magnitude = Math.abs(point.effect);
    if (currentPeak == null || magnitude > currentPeak.magnitude) {
      return { day: point.day, magnitude };
    }
    return currentPeak;
  }, null);
  if (peakPoint != null) {
    markers.push({ day: peakPoint.day, label: "peak" });
  }
  return Array.from(
    new Map(
      markers
        .filter((marker) => marker.day <= max)
        .sort((left, right) => left.day - right.day)
        .map((marker) => [marker.label, marker]),
    ).values(),
  );
}

function buildPhaseMarkers(mode: DagMode, phase: string): PhaseMarker[] | undefined {
  if (mode !== "rung3") {
    return undefined;
  }
  return [
    { position: 0.1, label: "abduction", active: phase === "abduction" },
    { position: 0.275, label: "action", active: phase === "surgery" },
    {
      position: 0.675,
      label: "prediction",
      active: phase === "prediction" || phase === "settled",
    },
  ];
}

export function buildInterventionDagViewModel(args: {
  constructs: Construct[];
  result: Stage6SimulationResult | null;
  animation: Pick<DagAnimationState, "phase" | "timeIndex" | "nodePhases" | "nodeEffects" | "abductedValues">;
}): InterventionDagViewModel {
  const { constructs, result, animation } = args;
  const mode = getDagMode(result);
  if (!result) {
    return {
      mode,
      actionDescription: null,
      evidenceDescription: null,
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
    evidenceDescription: formatEvidenceWindowLabel(result),
    timeStepsDays,
    temporalMarkers: buildTemporalMarkers(result, timeStepsDays),
    phaseMarkers: buildPhaseMarkers(mode, animation.phase),
    nodeData: Object.fromEntries(
      constructs.map((construct) => [
        construct.name,
        {
          rung: result.rung,
          animPhase: animation.nodePhases[construct.name] ?? "idle",
          effectMagnitude: animation.nodeEffects[construct.name] ?? null,
          abductedValue: animation.abductedValues[construct.name] ?? null,
          timeIndex: animation.timeIndex,
          timeStepsDays,
          effectTimeSeries: getNodeEffectSeries(result, construct.name),
          actionLabelShort,
          actionReferenceLabel,
        },
      ]),
    ),
  };
}
