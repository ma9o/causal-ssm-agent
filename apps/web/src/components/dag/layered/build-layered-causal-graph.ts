import type { CausalEdge, Construct, LatentStructure } from "@nof1-causal-lab/api-types";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
import { baseId, buildGhostLinks, splitEdgesWithGlyphs, unrollCausalLinks } from "../unroll";
import { causalEdgeKey } from "./layered-causal-graph-model";

export const LAYERED_NODE_WIDTH = 250;
export const LAYERED_NODE_HEIGHT = 132;
export const LAYERED_HISTORY_WIDTH = 156;
export const LAYERED_HISTORY_HEIGHT = 54;
export const LAYERED_EDGE_SLOT_WIDTH = 78;
export const LAYERED_EDGE_SLOT_HEIGHT = 32;

export type LayeredGraphNodeMeta =
  | { kind: "construct"; construct: Construct }
  | { kind: "history"; construct: Construct }
  | { kind: "edge_slot"; edgeId: string };

export interface LayeredGraphEdgeMeta {
  id: string;
  cause: string;
  effect: string;
  source: string;
  target: string;
  lagged: boolean;
  isSelf: boolean;
  slotId: string;
}

export interface LayeredGraphSegmentMeta {
  edgeId: string;
  markerEnd: boolean;
}

export interface LayeredGraphBundle {
  graph: DagGraphInput;
  nodeMeta: Map<string, LayeredGraphNodeMeta>;
  edgeMeta: Map<string, LayeredGraphEdgeMeta>;
  segmentMeta: Map<string, LayeredGraphSegmentMeta>;
}

const partition = (value: 0 | 1 | 2): Record<string, string> => ({
  "elk.partitioning.partition": String(value),
});

function constructPartition(construct: Construct): 0 | 2 {
  return construct.temporal_status === "time_invariant" ? 0 : 2;
}

/**
 * Build the permanent graph geometry exclusively from LatentStructure.
 * Every later artifact layer receives this same graph and can only decorate it.
 */
export function buildLayeredCausalGraph(structure: LatentStructure): LayeredGraphBundle {
  const constructByName = new Map(
    structure.constructs.map((construct) => [construct.name, construct] as const),
  );
  for (const edge of structure.edges) {
    if (!constructByName.has(edge.cause) || !constructByName.has(edge.effect)) {
      throw new Error(
        `Causal edge '${edge.cause}→${edge.effect}' references an unknown construct.`,
      );
    }
  }

  const timeVaryingNames = new Set(
    structure.constructs
      .filter((construct) => construct.temporal_status === "time_varying")
      .map((construct) => construct.name),
  );
  const selfDynamicConstructs = structure.constructs.filter(
    (construct) => construct.role === "endogenous" && construct.temporal_status === "time_varying",
  );
  const causalLinks = unrollCausalLinks(
    structure.edges.filter((edge) => edge.cause !== edge.effect),
    timeVaryingNames,
  );
  const selfLinks = buildGhostLinks(
    selfDynamicConstructs.map((construct) => ({
      from: construct.name,
      to: construct.name,
    })),
  );
  const ghosts = new Set([...causalLinks.ghosts, ...selfLinks.ghosts]);

  const edgeDefinitions: Array<Omit<LayeredGraphEdgeMeta, "slotId">> = [
    ...causalLinks.edges.map((edge) => ({
      id: causalEdgeKey(edge.cause, edge.effect, edge.lagged),
      cause: edge.cause,
      effect: edge.effect,
      source: edge.source,
      target: edge.target,
      lagged: edge.lagged,
      isSelf: false,
    })),
    ...selfLinks.edges.map((edge) => ({
      id: `self:${baseId(edge.source)}`,
      cause: baseId(edge.source),
      effect: baseId(edge.target),
      source: edge.source,
      target: edge.target,
      lagged: true,
      isSelf: true,
    })),
  ];

  const split = splitEdgesWithGlyphs(
    edgeDefinitions.map((edge) => ({
      a: edge.source,
      b: edge.target,
      isSelf: edge.isSelf,
      lagged: edge.lagged,
    })),
    { width: LAYERED_EDGE_SLOT_WIDTH, height: LAYERED_EDGE_SLOT_HEIGHT },
  );

  const nodeMeta = new Map<string, LayeredGraphNodeMeta>();
  const nodes: DagGraphInput["nodes"] = [];
  for (const construct of structure.constructs) {
    nodeMeta.set(construct.name, { kind: "construct", construct });
    nodes.push({
      id: construct.name,
      width: LAYERED_NODE_WIDTH,
      height: LAYERED_NODE_HEIGHT,
      layoutOptions: partition(constructPartition(construct)),
    });
  }
  for (const ghost of ghosts) {
    const construct = constructByName.get(baseId(ghost));
    if (!construct) {
      throw new Error(`Temporal copy '${ghost}' has no source construct.`);
    }
    nodeMeta.set(ghost, { kind: "history", construct });
    nodes.push({
      id: ghost,
      width: LAYERED_HISTORY_WIDTH,
      height: LAYERED_HISTORY_HEIGHT,
      layoutOptions: partition(1),
    });
  }

  const edgeMeta = new Map<string, LayeredGraphEdgeMeta>();
  const segmentMeta = new Map<string, LayeredGraphSegmentMeta>();
  split.glyphNodes.forEach((slot, index) => {
    const definition = edgeDefinitions[index];
    const sourceConstruct = constructByName.get(baseId(definition.source));
    const targetConstruct = constructByName.get(baseId(definition.target));
    if (!sourceConstruct || !targetConstruct) {
      throw new Error(`Edge slot '${slot.id}' has an unknown endpoint.`);
    }
    const sourcePartition = definition.source.endsWith("__p")
      ? 1
      : constructPartition(sourceConstruct);
    const targetPartition = constructPartition(targetConstruct);
    const slotPartition =
      sourcePartition < targetPartition
        ? Math.max(sourcePartition, targetPartition - 1)
        : sourcePartition;

    nodes.push({
      ...slot,
      layoutOptions: partition(slotPartition as 0 | 1 | 2),
    });
    nodeMeta.set(slot.id, { kind: "edge_slot", edgeId: definition.id });
    edgeMeta.set(definition.id, { ...definition, slotId: slot.id });
    segmentMeta.set(`e${index}s`, { edgeId: definition.id, markerEnd: false });
    segmentMeta.set(`e${index}t`, { edgeId: definition.id, markerEnd: true });
  });

  return {
    graph: {
      nodes,
      edges: split.edges,
      direction: "RIGHT",
      layoutOptions: {
        "elk.partitioning.activate": "true",
        "elk.layered.spacing.nodeNodeBetweenLayers": "54",
        "elk.spacing.nodeNode": "30",
        "elk.spacing.edgeNode": "24",
        "elk.spacing.edgeEdge": "14",
        "elk.layered.spacing.edgeNodeBetweenLayers": "26",
      },
    },
    nodeMeta,
    edgeMeta,
    segmentMeta,
  };
}

export function sourceEdgeKey(edge: CausalEdge): string {
  return causalEdgeKey(edge.cause, edge.effect, edge.lagged);
}
