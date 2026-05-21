import type { CausalEdge, Construct } from "@nof1-causal-lab/api-types";
import { useCallback, useEffect, useRef, useState } from "react";
import type {
  EdgeAnimState,
  NodeAnimPhase,
  Stage6SimulationResult,
} from "@/components/dag/intervention-dag-types";
import {
  getEffectTrajectoryDays,
  getNodeEffectSeries,
} from "@/components/dag/intervention-dag-semantics";

export interface DagAnimationConfig {
  edges: CausalEdge[];
  constructs: Construct[];
  result: Stage6SimulationResult;
  durationMs?: number;
}

export interface DagAnimationState {
  phase: string;
  progress: number;
  timeIndex: number;
  isPlaying: boolean;
  nodePhases: Record<string, NodeAnimPhase>;
  nodeEffects: Record<string, number>;
  edgeStates: Record<string, EdgeAnimState>;
  startStateValues: Record<string, number | null>;
}

export interface DagAnimationControls {
  play: () => void;
  pause: () => void;
  reset: () => void;
  scrubTo: (timeIndex: number) => void;
}

type GraphAnimationFrame = Pick<
  DagAnimationState,
  "nodePhases" | "nodeEffects" | "edgeStates" | "startStateValues"
>;

function getDownstreamNodes(treatment: string, edges: CausalEdge[]): string[] {
  const children = new Map<string, string[]>();
  for (const edge of edges) {
    const list = children.get(edge.cause) ?? [];
    list.push(edge.effect);
    children.set(edge.cause, list);
  }
  const order: string[] = [];
  const visited = new Set<string>();
  const queue = [treatment];
  while (queue.length > 0) {
    const node = queue.shift();
    if (!node || visited.has(node)) {
      continue;
    }
    visited.add(node);
    order.push(node);
    for (const child of children.get(node) ?? []) {
      queue.push(child);
    }
  }
  return order;
}

function edgeKey(cause: string, effect: string): string {
  return `${cause}\u2192${effect}`;
}

function incomingKeys(target: string, edges: CausalEdge[]): string[] {
  return edges
    .filter((edge) => edge.effect === target)
    .map((edge) => edgeKey(edge.cause, edge.effect));
}

function outgoingKeys(source: string, edges: CausalEdge[]): string[] {
  return edges
    .filter((edge) => edge.cause === source)
    .map((edge) => edgeKey(edge.cause, edge.effect));
}

function nodeEffectAt(
  result: Stage6SimulationResult,
  nodeName: string,
  timeIndex: number,
): number {
  return getNodeEffectSeries(result, nodeName)?.[timeIndex] ?? 0;
}

function createEmptyState(
  constructs: Construct[],
  edges: CausalEdge[],
): GraphAnimationFrame {
  const nodePhases: Record<string, NodeAnimPhase> = {};
  const nodeEffects: Record<string, number> = {};
  const startStateValues: Record<string, number | null> = {};
  for (const construct of constructs) {
    nodePhases[construct.name] = "idle";
    nodeEffects[construct.name] = 0;
    startStateValues[construct.name] = null;
  }

  const edgeStates: Record<string, EdgeAnimState> = {};
  for (const edge of edges) {
    edgeStates[edgeKey(edge.cause, edge.effect)] = "normal";
  }
  return { nodePhases, nodeEffects, edgeStates, startStateValues };
}

function clampTreatmentNode(
  frame: GraphAnimationFrame,
  result: Stage6SimulationResult,
  treatment: string,
  timeIndex: number,
): void {
  frame.nodePhases[treatment] = "clamped";
  frame.nodeEffects[treatment] = nodeEffectAt(result, treatment, timeIndex);
}

function cutIncomingEdges(frame: GraphAnimationFrame, incoming: Set<string>): void {
  for (const key of incoming) {
    frame.edgeStates[key] = "cut";
  }
}

function revealStartStateNodes(
  frame: GraphAnimationFrame,
  nodeNames: string[],
  startState: Record<string, number>,
  showCount = nodeNames.length,
): void {
  for (let index = 0; index < Math.min(showCount, nodeNames.length); index++) {
    const name = nodeNames[index];
    if (!name) {
      continue;
    }
    frame.nodePhases[name] = "start_state";
    frame.startStateValues[name] = startState[name] ?? null;
  }
}

function applyNodeEffects(
  frame: GraphAnimationFrame,
  result: Stage6SimulationResult,
  nodeNames: string[],
  timeIndex: number,
): void {
  for (const nodeName of nodeNames) {
    const effect = nodeEffectAt(result, nodeName, timeIndex);
    frame.nodeEffects[nodeName] = effect;
    frame.nodePhases[nodeName] = "active";
  }
}

function dimOutsideCausalCone(
  frame: GraphAnimationFrame,
  constructs: Construct[],
  downstreamNodes: Set<string>,
): void {
  for (const construct of constructs) {
    if (downstreamNodes.has(construct.name)) {
      continue;
    }
    frame.nodePhases[construct.name] = "dimmed";
  }
}

function applyEdgeFlowStates(
  frame: GraphAnimationFrame,
  incoming: Set<string>,
  downstreamKeys: Set<string>,
  options?: {
    dimOutsideDownstream?: boolean;
  },
): void {
  const dimOutsideDownstream = options?.dimOutsideDownstream ?? false;
  for (const key of Object.keys(frame.edgeStates)) {
    if (incoming.has(key)) {
      continue;
    }
    if (!downstreamKeys.has(key)) {
      if (dimOutsideDownstream) {
        frame.edgeStates[key] = "dimmed";
      }
      continue;
    }
    frame.edgeStates[key] = "flowing";
  }
}

function deriveRung2(
  progress: number,
  config: DagAnimationConfig,
  downstream: string[],
): Omit<DagAnimationState, "isPlaying"> {
  const treatment = config.result.action.variable;
  const downstreamNodes = new Set(downstream);
  const incoming = new Set(incomingKeys(treatment, config.edges));
  const downstreamKeys = new Set(downstream.flatMap((node) => outgoingKeys(node, config.edges)));
  const base = createEmptyState(config.constructs, config.edges);
  const timelineDays = getEffectTrajectoryDays(config.result);
  let phase = "idle";
  let timeIndex = 0;

  if (progress < 0.1) {
    phase = "clamping";
    clampTreatmentNode(base, config.result, treatment, 0);
  } else if (progress < 0.25) {
    phase = "surgery";
    clampTreatmentNode(base, config.result, treatment, 0);
    cutIncomingEdges(base, incoming);
  } else if (timelineDays.length > 0) {
    const prop = (progress - 0.25) / 0.75;
    const maxIdx = timelineDays.length - 1;
    timeIndex = Math.min(maxIdx, Math.round(prop * maxIdx));
    phase = timeIndex >= maxIdx ? "settled" : "propagating";
    clampTreatmentNode(base, config.result, treatment, timeIndex);
    cutIncomingEdges(base, incoming);
    applyNodeEffects(
      base,
      config.result,
      downstream.filter((node) => node !== treatment),
      timeIndex,
    );
    dimOutsideCausalCone(base, config.constructs, downstreamNodes);
    applyEdgeFlowStates(base, incoming, downstreamKeys, {
      dimOutsideDownstream: true,
    });
  }

  return {
    phase,
    progress,
    timeIndex,
    ...base,
  };
}

function deriveRung3(
  progress: number,
  config: DagAnimationConfig,
  downstream: string[],
): Omit<DagAnimationState, "isPlaying"> {
  const treatment = config.result.action.variable;
  const downstreamNodes = new Set(downstream);
  const incoming = new Set(incomingKeys(treatment, config.edges));
  const downstreamKeys = new Set(downstream.flatMap((node) => outgoingKeys(node, config.edges)));
  const base = createEmptyState(config.constructs, config.edges);
  const timelineDays = getEffectTrajectoryDays(config.result);
  const startState = (config.result.visualization?.start_state ?? {}) as Record<
    string,
    number
  >;
  let phase = "idle";
  let timeIndex = 0;

  if (progress < 0.2) {
    phase = "start_state";
    revealStartStateNodes(
      base,
      config.constructs.map((construct) => construct.name),
      startState,
      Math.ceil((progress / 0.2) * config.constructs.length),
    );
  } else if (progress < 0.35) {
    phase = "surgery";
    clampTreatmentNode(base, config.result, treatment, 0);
    revealStartStateNodes(
      base,
      config.constructs
        .map((construct) => construct.name)
        .filter((name) => name !== treatment),
      startState,
    );
    cutIncomingEdges(base, incoming);
  } else if (timelineDays.length > 0) {
    const prop = (progress - 0.35) / 0.65;
    const maxIdx = timelineDays.length - 1;
    timeIndex = Math.min(maxIdx, Math.round(prop * maxIdx));
    phase = timeIndex >= maxIdx ? "settled" : "prediction";
    clampTreatmentNode(base, config.result, treatment, timeIndex);
    cutIncomingEdges(base, incoming);
    applyNodeEffects(
      base,
      config.result,
      config.constructs
        .map((construct) => construct.name)
        .filter((name) => name !== treatment),
      timeIndex,
    );
    dimOutsideCausalCone(base, config.constructs, downstreamNodes);
    applyEdgeFlowStates(base, incoming, downstreamKeys, {
      dimOutsideDownstream: true,
    });
  }

  return {
    phase,
    progress,
    timeIndex,
    ...base,
  };
}

export function deriveDagAnimationFrame(
  progress: number,
  config: DagAnimationConfig | null,
): Omit<DagAnimationState, "isPlaying"> | null {
  if (!config) {
    return null;
  }

  const downstream = getDownstreamNodes(config.result.action.variable, config.edges);
  return config.result.rung === 3
    ? deriveRung3(progress, config, downstream)
    : deriveRung2(progress, config, downstream);
}

export function useDagAnimation(
  config: DagAnimationConfig | null,
): DagAnimationState & DagAnimationControls {
  const [progress, setProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(() => config != null);
  const rafRef = useRef(0);
  const startTimeRef = useRef(0);
  const startProgressRef = useRef(0);

  const duration = config?.durationMs ?? 8000;

  useEffect(() => {
    if (!isPlaying || !config) {
      return;
    }

    startTimeRef.current = performance.now();
    startProgressRef.current = progress;

    const tick = (now: number) => {
      const elapsed = now - startTimeRef.current;
      const next = Math.min(1, startProgressRef.current + elapsed / duration);
      setProgress(next);
      if (next >= 1) {
        setIsPlaying(false);
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [config, duration, isPlaying, progress]);

  const play = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);
  const reset = useCallback(() => {
    setIsPlaying(false);
    setProgress(0);
  }, []);

  const scrubTo = useCallback(
    (timeIndex: number) => {
      if (!config) {
        return;
      }
      setIsPlaying(false);
      const steps = getEffectTrajectoryDays(config.result);
      const maxIdx = steps.length - 1;
      if (maxIdx <= 0) {
        return;
      }
      const propStart = config.result.rung === 3 ? 0.35 : 0.25;
      const propRange = config.result.rung === 3 ? 0.65 : 0.75;
      setProgress(Math.min(1, propStart + (timeIndex / maxIdx) * propRange));
    },
    [config],
  );

  const derived = deriveDagAnimationFrame(progress, config);

  if (!derived) {
    return {
      phase: "idle",
      progress: 0,
      timeIndex: 0,
      isPlaying: false,
      nodePhases: {},
      nodeEffects: {},
      edgeStates: {},
      startStateValues: {},
      play,
      pause,
      reset,
      scrubTo,
    };
  }

  return { ...derived, isPlaying, play, pause, reset, scrubTo };
}
