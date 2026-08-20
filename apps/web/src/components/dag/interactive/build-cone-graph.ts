import type { DagDirection, DagGraphInput } from "@/lib/utils/dag-graph-layout";
import type { CausalEdge, Construct, Indicator } from "@nof1-causal-lab/api-types";
import { baseId, buildGhostLinks, unrollCausalLinks } from "../unroll";

// Re-exported so existing importers (e.g. interactive-dag) keep their import path
// while the convention itself lives in the shared ../unroll module.
export { baseId };

export const CARD_W = 252;
export const CARD_H = 152;
export const HISTORY_H = 64;
export const MINI_H = 22;
export const IGAP = 6;
export const ISTACK_TOP = 16;

/** ELK spacing matching the playground's `elkOptions`. */
const LAYOUT_OPTIONS: Record<string, string> = {
  "elk.layered.spacing.nodeNodeBetweenLayers": "56",
  "elk.spacing.nodeNode": "30",
  "elk.spacing.edgeNode": "28",
  "elk.spacing.edgeEdge": "16",
  "elk.layered.spacing.edgeNodeBetweenLayers": "28",
};

export interface SimulationGraph {
  graph: DagGraphInput;
  edgeMeta: Map<string, { a: string; b: string; isSelf: boolean; lagged: boolean }>;
}

/**
 * Build the scientific DAG directly from backend-declared constructs and edges.
 * Lagged cross-construct effects originate at t−1; contemporaneous effects stay
 * within t. A fitted state gets a separate persistence edge only when an
 * `ar_coefficient` posterior exists.
 */
export function buildSimulationGraph(
  constructs: Construct[],
  edges: CausalEdge[],
  opts: {
    dir: DagDirection;
    showIndicators: boolean;
    showUnroll: boolean;
    indicators: Indicator[];
    /** Fitted states with a materialized daily-persistence posterior. */
    persistenceNodes: string[];
  },
): SimulationGraph {
  const present = new Set(constructs.map((c) => c.name));
  const timeVaryingNames = new Set(
    constructs
      .filter((construct) => construct.temporal_status === "time_varying")
      .map((construct) => construct.name),
  );
  const indCount = (node: string) =>
    opts.indicators.filter((ind) => ind.construct_name === node).length;

  const selfTap = opts.persistenceNodes.filter((name) => present.has(name));
  // A fitted persistence parameter materializes a t−1 → t self edge without
  // inventing the richer drift decomposition that simulation results do not expose.
  const selfLinks = opts.showUnroll
    ? buildGhostLinks(selfTap.map((id) => ({ from: id, to: id })))
    : { ghosts: [] as string[], edges: [] as { source: string; target: string }[] };
  const causalLinks = unrollCausalLinks(
    edges.filter(
      (edge) => edge.cause !== edge.effect && present.has(edge.cause) && present.has(edge.effect),
    ),
    opts.showUnroll ? timeVaryingNames : new Set<string>(),
  );
  const ghosts = new Set([...causalLinks.ghosts, ...selfLinks.ghosts]);

  const sizeOf = (id: string): { w: number; h: number } => {
    const base = baseId(id);
    if (id !== base) {
      return { w: CARD_W, h: HISTORY_H };
    }
    if (opts.showIndicators && id === base && indCount(base) > 0) {
      return { w: CARD_W, h: CARD_H + ISTACK_TOP + indCount(base) * (MINI_H + IGAP) };
    }
    return { w: CARD_W, h: CARD_H };
  };

  const nodes: DagGraphInput["nodes"] = [];
  for (const c of constructs) {
    const s = sizeOf(c.name);
    nodes.push({ id: c.name, width: s.w, height: s.h });
  }
  for (const g of ghosts) {
    nodes.push({ id: g, width: CARD_W, height: HISTORY_H });
  }

  const pairs: { a: string; b: string; lagged: boolean }[] = [];
  for (const edge of causalLinks.edges) {
    pairs.push({ a: edge.source, b: edge.target, lagged: edge.lagged });
  }
  for (const link of selfLinks.edges) {
    pairs.push({ a: link.source, b: link.target, lagged: true });
  }

  const edgeMeta = new Map<string, { a: string; b: string; isSelf: boolean; lagged: boolean }>();
  const elkEdges: DagGraphInput["edges"] = [];
  pairs.forEach(({ a, b, lagged }, i) => {
    const id = `e${i}`;
    edgeMeta.set(id, { a, b, isSelf: baseId(a) === baseId(b), lagged });
    elkEdges.push({ id, source: a, target: b });
  });

  return {
    graph: { nodes, edges: elkEdges, direction: opts.dir, layoutOptions: LAYOUT_OPTIONS },
    edgeMeta,
  };
}
