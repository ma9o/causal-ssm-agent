import ELK, { type ElkNode } from "elkjs/lib/elk.bundled.js";

/**
 * Generic ELK layout for the bespoke DAG renderer (shared by the static
 * structure DAG and the interactive simulation DAG).
 *
 * Unlike the legacy `dag-layout.ts` — which used ELK only to *place* nodes and
 * then discarded the routing in favour of React Flow's smoothstep edges — this
 * returns ELK's ORTHOGONAL edge routing (start + bend points + end) so the
 * renderer can draw the lines itself. Input is a plain `{nodes, edges}` graph,
 * not the construct/edge domain types, so callers build whatever graph they
 * need (structure graph, or the unrolled / glyph-split simulation graph) and
 * feed it through the same layout + render path.
 */

export interface Point {
  x: number;
  y: number;
}

export type DagDirection = "RIGHT" | "DOWN" | "LEFT" | "UP";

export interface DagLayoutNodeInput {
  id: string;
  width: number;
  height: number;
  /** Per-node ELK options (e.g. partitions for the interactive glyph columns). */
  layoutOptions?: Record<string, string>;
}

export interface DagLayoutEdgeInput {
  id: string;
  source: string;
  target: string;
}

export interface DagGraphInput {
  nodes: DagLayoutNodeInput[];
  edges: DagLayoutEdgeInput[];
  /** Layout flow direction. Defaults to RIGHT (cause → effect, left to right). */
  direction?: DagDirection;
  /** Graph-level ELK option overrides, merged over the defaults. */
  layoutOptions?: Record<string, string>;
}

export interface DagLayoutNode {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DagLayoutEdge {
  id: string;
  source: string;
  target: string;
  /** Routed polyline: [startPoint, ...bendPoints, endPoint]. */
  points: Point[];
}

export interface DagLayoutResult {
  width: number;
  height: number;
  nodes: DagLayoutNode[];
  edges: DagLayoutEdge[];
}

export const EMPTY_LAYOUT: DagLayoutResult = {
  width: 0,
  height: 0,
  nodes: [],
  edges: [],
};

const DEFAULT_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.layering.strategy": "NETWORK_SIMPLEX",
  "elk.layered.spacing.nodeNodeBetweenLayers": "64",
  "elk.spacing.nodeNode": "36",
  "elk.spacing.edgeNode": "28",
  "elk.spacing.edgeEdge": "16",
  "elk.layered.spacing.edgeNodeBetweenLayers": "28",
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
};

/** Build the ELK graph from a generic graph input. Pure — no layout is run. */
export function buildElkGraph(input: DagGraphInput): ElkNode {
  return {
    id: "root",
    layoutOptions: {
      ...DEFAULT_OPTIONS,
      "elk.direction": input.direction ?? "RIGHT",
      ...(input.layoutOptions ?? {}),
    },
    children: input.nodes.map((n) => ({
      id: n.id,
      width: n.width,
      height: n.height,
      ...(n.layoutOptions ? { layoutOptions: n.layoutOptions } : {}),
    })),
    edges: input.edges.map((e) => ({
      id: e.id,
      sources: [e.source],
      targets: [e.target],
    })),
  };
}

/** Read a laid-out ELK graph into the renderer's flat geometry. Pure. */
export function readElkLayout(root: ElkNode): DagLayoutResult {
  const nodes: DagLayoutNode[] = (root.children ?? []).map((c) => ({
    id: c.id,
    x: c.x ?? 0,
    y: c.y ?? 0,
    width: c.width ?? 0,
    height: c.height ?? 0,
  }));

  const edges: DagLayoutEdge[] = (root.edges ?? []).map((e) => {
    const section = e.sections?.[0];
    const points: Point[] = section
      ? [section.startPoint, ...(section.bendPoints ?? []), section.endPoint].map((p) => ({
          x: p.x,
          y: p.y,
        }))
      : [];
    return {
      id: e.id ?? "",
      source: e.sources?.[0] ?? "",
      target: e.targets?.[0] ?? "",
      points,
    };
  });

  return {
    width: root.width ?? 0,
    height: root.height ?? 0,
    nodes,
    edges,
  };
}

const elk = new ELK();

/** Run ELK layout and return the renderer's flat, routed geometry. */
export async function runDagLayout(input: DagGraphInput): Promise<DagLayoutResult> {
  const laidOut = await elk.layout(buildElkGraph(input));
  return readElkLayout(laidOut);
}
