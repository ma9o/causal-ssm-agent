import type { DagDirection, DagGraphInput } from "@/lib/utils/dag-graph-layout";
import type { CausalEdge, Construct, Indicator } from "@nof1-causal-lab/api-types";
import { baseId, buildGhostLinks } from "../unroll";

// Re-exported so existing importers (e.g. interactive-dag) keep their import path
// while the convention itself lives in the shared ../unroll module.
export { baseId };

export const CARD_W = 252;
export const CARD_H = 152;
export const GLYPH_W = 86;
export const GLYPH_H = 36;
export const MINI_H = 34;
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

export interface GlyphGraph {
  graph: DagGraphInput;
  /** Glyph layout-node id → the edge it sits on. */
  glyphs: Map<string, { a: string; b: string; isSelf: boolean; lagged: boolean }>;
}

/**
 * Build the ELK graph for the interactive DAG, exactly as the playground does:
 * each causal edge a→b is split into a→[glyph]→b with the drift glyph as a real
 * layout node on the edge; every endo latent gets a faded t−1 ghost + a
 * self-edge carrying its self-effect glyph. Cards grow to reserve indicator-stack
 * space when toggled on. (Partitions are omitted — lagged edges can form cycles,
 * which ELK's layered algorithm breaks on its own.)
 *
 * Renders the full projected estimation graph: every passed-in construct (the
 * retained latent states plus the known-input drivers) and every edge among
 * them. Marginalized confounders and non-identifiable nodes are already excluded
 * upstream by the estimation projection, so no further filtering happens here —
 * held drivers that don't reach the outcome stay visible as disconnected cards
 * rather than silently disappearing.
 */
export function buildGlyphGraph(
  constructs: Construct[],
  edges: CausalEdge[],
  opts: {
    dir: DagDirection;
    showIndicators: boolean;
    showUnroll: boolean;
    indicators: Indicator[];
  },
): GlyphGraph {
  const present = new Set(constructs.map((c) => c.name));
  const isEndo = new Map(constructs.map((c) => [c.name, c.role === "endogenous"]));
  const indCount = (node: string) =>
    opts.indicators.filter((ind) => ind.construct_name === node).length;

  const selfTap = constructs.filter((c) => isEndo.get(c.name)).map((c) => c.name);
  // Each endo latent's NodePotential self-dynamics: its faded t−1 ghost feeds its
  // present-time self. Same unrolling convention as the static structure DAG.
  const selfLinks = opts.showUnroll
    ? buildGhostLinks(selfTap.map((id) => ({ from: id, to: id })))
    : { ghosts: [] as string[], edges: [] as { source: string; target: string }[] };

  const sizeOf = (id: string): { w: number; h: number } => {
    const base = baseId(id);
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
  for (const g of selfLinks.ghosts) {
    nodes.push({ id: g, width: CARD_W, height: CARD_H });
  }

  const pairs: { a: string; b: string; lagged: boolean }[] = [];
  for (const e of edges) {
    if (e.cause === e.effect || !present.has(e.cause) || !present.has(e.effect)) continue;
    pairs.push({ a: e.cause, b: e.effect, lagged: e.lagged });
  }
  for (const link of selfLinks.edges) {
    pairs.push({ a: link.source, b: link.target, lagged: false });
  }

  const glyphs = new Map<string, { a: string; b: string; isSelf: boolean; lagged: boolean }>();
  const elkEdges: DagGraphInput["edges"] = [];
  pairs.forEach(({ a, b, lagged }, i) => {
    const gid = `G__${i}`;
    nodes.push({ id: gid, width: GLYPH_W, height: GLYPH_H });
    glyphs.set(gid, { a, b, isSelf: baseId(a) === baseId(b), lagged });
    elkEdges.push({ id: `e${i}s`, source: a, target: gid });
    elkEdges.push({ id: `e${i}t`, source: gid, target: b });
  });

  return {
    graph: { nodes, edges: elkEdges, direction: opts.dir, layoutOptions: LAYOUT_OPTIONS },
    glyphs,
  };
}
