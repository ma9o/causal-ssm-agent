/**
 * Time-unrolling for the causal-DAG renderers (shared by the static structure
 * DAG and the analysis interactive DAG).
 *
 * A lagged dependency — something at t−1 driving an effect at t — is drawn by
 * *unrolling time* rather than by tagging the edge: the t−1 source becomes a
 * faded "ghost" copy of the present node and the edge runs ghost → present. The
 * present and past slices look identical, the past just fainter. This mirrors
 * the 2-timestep unrolling the backend uses for identification (A3a in
 * docs/modeling/assumptions.md), so the picture matches the modeling semantics.
 *
 * This module owns only the cross-consumer *convention* (the `__p` id suffix,
 * the ghost fade, and the ghost→present link builder). It is deliberately NOT
 * in `core/`: the core renderer is domain-agnostic (plain `{nodes, edges}`),
 * whereas which links are temporal is a domain decision each consumer makes
 * (lagged structural edges here; NodePotential self-dynamics in analysis).
 */

/** Suffix marking a node id as the t−1 (previous-timestep) ghost of its base. */
export const GHOST_SUFFIX = "__p";

/** Opacity for t−1 ghost cards — present-time looks identical, just fainter. */
export const GHOST_OPACITY = 0.42;

/** The t−1 ghost id for a present-time node. */
export const ghostId = (base: string): string => `${base}${GHOST_SUFFIX}`;

/** Strip the ghost suffix to recover the present-time construct name. */
export const baseId = (id: string): string =>
  id.endsWith(GHOST_SUFFIX) ? id.slice(0, -GHOST_SUFFIX.length) : id;

/** Whether an id refers to a t−1 ghost rather than a present-time node. */
export const isGhost = (id: string): boolean => id.endsWith(GHOST_SUFFIX);

/** A t−1 → t link: `from` is taken at the previous step (routed from its ghost). */
export interface TemporalLink {
  from: string;
  to: string;
}

export interface GhostLinks {
  /** Distinct ghost node ids to add to the t−1 slice. */
  ghosts: string[];
  /** Edges routed from each ghost into its present-time target. */
  edges: { source: string; target: string }[];
}

/**
 * Turn temporal links into the ghost nodes + ghost→present edges that realize
 * the unrolling. Callers add `ghosts` to their node list (sized + faded) and
 * `edges` alongside the contemporaneous edges.
 */
export function buildGhostLinks(links: TemporalLink[]): GhostLinks {
  const ghosts = new Set<string>();
  const edges = links.map((link) => {
    const source = ghostId(link.from);
    ghosts.add(source);
    return { source, target: link.to };
  });
  return { ghosts: [...ghosts], edges };
}

/** Glyph/spacer node size — matches the analysis interactive DAG's drift glyphs. */
export const GLYPH_W = 86;
export const GLYPH_H = 36;

/**
 * ELK spacing shared with the analysis interactive DAG so the structural and
 * intervention graphs lay out identically given the same node/edge structure.
 * (build-cone-graph keeps a matching local copy for now; unify when that path
 * is refactored onto this module.)
 */
export const DAG_LAYOUT_OPTIONS: Record<string, string> = {
  "elk.layered.spacing.nodeNodeBetweenLayers": "56",
  "elk.spacing.nodeNode": "30",
  "elk.spacing.edgeNode": "28",
  "elk.spacing.edgeEdge": "16",
  "elk.layered.spacing.edgeNodeBetweenLayers": "28",
};

/** A causal pair to lay out as `a → [glyph] → b`. */
export interface GlyphPair {
  a: string;
  b: string;
  /** Self-dynamics edge (ghost → its own present node) rather than a cross-edge. */
  isSelf: boolean;
  lagged: boolean;
}

export interface GlyphSplit {
  glyphNodes: { id: string; width: number; height: number }[];
  edges: { id: string; source: string; target: string }[];
  /** Glyph layout-node id (`G__<i>`) → the pair it sits on. */
  glyphs: Map<string, GlyphPair>;
}

/**
 * Split each pair `a → b` into `a → [glyph] → b`, inserting a glyph node on the
 * edge — the same construction the analysis DAG uses. Putting a real node on every
 * edge is what gives the layered layout its column rhythm (each edge spans an
 * extra layer), so structural and intervention graphs share it.
 *
 * `glyphSize` controls that node's footprint. analysis uses the full GLYPH_W×GLYPH_H
 * because it draws a drift glyph there, and the box hides ELK's port-entry jog. The
 * structural DAG leaves the slot empty and draws the edge straight through (it
 * concatenates the `e<i>s` + `e<i>t` routed segments), so it passes a near-point
 * size: layer insertion is size-independent, so the columns still match analysis,
 * but a point has no port-entry jog to expose — no squiggle.
 */
export function splitEdgesWithGlyphs(
  pairs: GlyphPair[],
  glyphSize: { width: number; height: number } = { width: GLYPH_W, height: GLYPH_H },
): GlyphSplit {
  const glyphNodes: GlyphSplit["glyphNodes"] = [];
  const edges: GlyphSplit["edges"] = [];
  const glyphs = new Map<string, GlyphPair>();
  pairs.forEach((pair, i) => {
    const gid = `G__${i}`;
    glyphNodes.push({ id: gid, width: glyphSize.width, height: glyphSize.height });
    glyphs.set(gid, pair);
    edges.push({ id: `e${i}s`, source: pair.a, target: gid });
    edges.push({ id: `e${i}t`, source: gid, target: pair.b });
  });
  return { glyphNodes, edges, glyphs };
}
