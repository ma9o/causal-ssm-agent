"use client";

import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import type { DagDirection, DagGraphInput, Point } from "@/lib/utils/dag-graph-layout";
import type { CausalEdge, Construct, Indicator, KnownInput } from "@nof1-causal-lab/api-types";
import { useCallback, useMemo, useState } from "react";
import { DagCanvasFrame, DagSvg } from "./core/dag-canvas";
import { DagDirectionToggle } from "./core/dag-direction-toggle";
import { DagEdge } from "./core/dag-edge";
import { DagNodeShell } from "./core/dag-node";
import { DagZoomControls } from "./core/dag-zoom-controls";
import {
  baseId,
  buildGhostLinks,
  DAG_LAYOUT_OPTIONS,
  GHOST_OPACITY,
  type GlyphPair,
  isGhost,
  splitEdgesWithGlyphs,
  unrollCausalLinks,
} from "./unroll";

export type ConstructStatus = "observed" | "marginalized" | "blocking";

interface StructureDagProps {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  /** Constructs compiled as observed transition inputs rather than latent states. */
  knownInputs?: KnownInput[];
  /**
   * Per-construct backend disposition/identifiability status. Colors the node border and any
   * incident edge: blocking → destructive, marginalized → warning.
   */
  nodeStatuses?: Record<string, ConstructStatus>;
  onNodeClick?: (constructName: string) => void;
  /** Initial layout flow direction. Defaults to cause → effect, left to right. */
  direction?: "RIGHT" | "DOWN";
}

// ---- node geometry — shared between the layout sizing and the SVG card ----
const NODE_W = 220;
const NODE_W_WITH_INDICATORS = 248;
const HEADER_H = 56; // name + role·temporal subtitle, no indicators
const SEPARATOR_Y = 46; // dashed rule between header and the indicator rows
const INDICATOR_ROW_H = 17;

const ZMIN = 0.4;
const ZMAX = 2.5;

function nodeHeight(indicatorCount: number): number {
  if (indicatorCount === 0) return HEADER_H;
  return SEPARATOR_Y + indicatorCount * INDICATOR_ROW_H + 8;
}

// Cross-construct lagged edges and self-dynamics originate from t−1 ghosts;
// contemporaneous edges stay within the present-time slice.
const EDGE_DEFAULT = "var(--edge-contemporary)";
const EDGE_BLOCKING = "var(--destructive)";
const EDGE_MARGINALIZED = "var(--warning)";
const SELF_EDGE_OPACITY = 0.45;

/** Edge tint from its endpoints' identifiability status (blocking wins over marginalized). */
function edgeStatusColor(
  a: ConstructStatus | undefined,
  b: ConstructStatus | undefined,
): { color: string; flagged: boolean } {
  if (a === "blocking" || b === "blocking") return { color: EDGE_BLOCKING, flagged: true };
  if (a === "marginalized" || b === "marginalized")
    return { color: EDGE_MARGINALIZED, flagged: true };
  return { color: EDGE_DEFAULT, flagged: false };
}

/**
 * Stitch the two glyph-split halves into one polyline and remove the spacer spur.
 * Splitting a → [spacer] → b makes ELK route into the spacer's ports; on a back-edge
 * it enters and exits the same side, leaving a short out-and-back "spur" by the
 * spacer (an empty slot would otherwise expose it as a squiggle). A spur is a
 * H-V-H (or V-H-V) run whose outer two segments
 * reverse direction with only a short perpendicular hop between — that overshoot is
 * dropped. Genuine routing detours (large perpendicular travel to clear a node) are
 * left untouched.
 */
function cleanGlyphPath(raw: Point[]): Point[] {
  const SPUR_PERP = 32;
  const dedupe = (pts: Point[]): Point[] =>
    pts.filter((p, i, a) => i === 0 || p.x !== a[i - 1].x || p.y !== a[i - 1].y);
  const pts = dedupe(raw.map((p) => ({ x: Math.round(p.x), y: Math.round(p.y) })));
  for (let i = 0; i + 3 < pts.length; ) {
    const a = pts[i];
    const b = pts[i + 1];
    const c = pts[i + 2];
    const d = pts[i + 3];
    const s1x = b.x - a.x;
    const s1y = b.y - a.y;
    const s2x = c.x - b.x;
    const s2y = c.y - b.y;
    const s3x = d.x - c.x;
    const s3y = d.y - c.y;
    if (s1y === 0 && s3y === 0 && s2x === 0 && s1x * s3x < 0 && Math.abs(s2y) <= SPUR_PERP) {
      pts.splice(i + 1, 2, { x: d.x, y: a.y }); // horizontal overshoot → clamp back
      i = Math.max(0, i - 1);
    } else if (s1x === 0 && s3x === 0 && s2y === 0 && s1y * s3y < 0 && Math.abs(s2x) <= SPUR_PERP) {
      pts.splice(i + 1, 2, { x: a.x, y: d.y }); // vertical overshoot → clamp back
      i = Math.max(0, i - 1);
    } else {
      i++;
    }
  }
  return dedupe(pts);
}

const labelize = (text: string): string => text.replace(/_/g, " ");

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, Math.max(1, max - 1))}…` : text;
}

/** Border accent: status wins, then selection/hover, else the default border. */
function nodeAccent(status: ConstructStatus | undefined, lit: boolean): string | undefined {
  if (status === "blocking") return "var(--destructive)";
  if (status === "marginalized") return "var(--warning)";
  if (lit) return "var(--primary)";
  return undefined;
}

interface StructureNodeProps {
  width: number;
  height: number;
  construct: Construct;
  indicators: Indicator[];
  knownInput?: KnownInput;
  status?: ConstructStatus;
  lit: boolean;
  /** Rendered as the faded t−1 ghost (no indicators, no status). */
  isPrev: boolean;
}

/** One construct card: name + role·temporal subtitle + its indicator rows. */
function StructureNode({
  width,
  height,
  construct,
  indicators,
  knownInput,
  status,
  lit,
  isPrev,
}: StructureNodeProps) {
  const isExo = construct.role === "exogenous";
  const vary = construct.temporal_status === "time_varying" ? "varying" : "invariant";
  const subtitle = `${isExo ? "theory exo" : "theory endo"} · ${vary}${knownInput ? " · known input" : isExo ? " · held" : ""}`;

  const reserved = (construct.is_outcome ? 44 : 28) + (isPrev ? 36 : 0);
  const title = `${construct.is_outcome ? "★ " : ""}${truncate(labelize(construct.name), Math.floor((width - reserved) / 6.6))}${isPrev ? " · t−1" : ""}`;
  const indNameMax = Math.floor((width - 72) / 5.8);
  const showIndicators = !isPrev && indicators.length > 0;

  return (
    <DagNodeShell
      width={width}
      height={height}
      title={title}
      subtitle={subtitle}
      accent={nodeAccent(isPrev ? undefined : status, lit)}
      highlighted={lit}
      outcome={construct.is_outcome}
    >
      <title>{construct.name}</title>
      {showIndicators ? (
        <>
          <line
            x1={12}
            y1={SEPARATOR_Y}
            x2={width - 12}
            y2={SEPARATOR_Y}
            stroke="var(--border)"
            strokeDasharray="3,3"
          />
          {indicators.map((ind, i) => {
            const y = SEPARATOR_Y + 14 + i * INDICATOR_ROW_H;
            return (
              <g key={ind.name}>
                <text x={14} y={y} fontSize={10} fill="var(--muted-foreground)">
                  {truncate(ind.name, indNameMax)}
                </text>
                <text
                  x={width - 12}
                  y={y}
                  fontSize={8.5}
                  textAnchor="end"
                  fill="var(--muted-foreground)"
                >
                  {ind.measurement_dtype}
                </text>
              </g>
            );
          })}
        </>
      ) : null}
    </DagNodeShell>
  );
}

function LegendSwatch({ border, faded }: { border?: string; faded?: boolean }) {
  return (
    <span
      className="inline-block h-3 w-3 rounded-sm border-2 bg-card align-[-2px]"
      style={{ borderColor: border ?? "var(--foreground)", opacity: faded ? GHOST_OPACITY : 1 }}
    />
  );
}

/**
 * Static causal-structure DAG for Stages 1a/1b. Uses the bespoke core's node /
 * edge primitives (`DagNodeShell`, `DagEdge`, ELK routing) on the same scrollable,
 * content-sized canvas as the analysis interactive DAG — full-size cards you scroll
 * and zoom, not a fit-to-box viewport — so the structural and intervention stages
 * read identically.
 *
 * It shows every signal the old React Flow `CausalDag` did: construct cards
 * (name, role·temporal, outcome ★, indicator rows, status-colored borders),
 * status-colored edges (red touching a blocking node, amber touching a marginalized
 * one), click-to-focus (highlight a node's neighborhood, dim the rest) with
 * `onNodeClick`, and edge hover.
 *
 * Topology AND layout mirror the analysis interactive DAG (the reference "good
 * state"): the ELK graph is built the same way — every causal edge is split
 * a → [glyph] → b (the glyph node per edge gives the layered layout its column
 * rhythm; see `splitEdgesWithGlyphs`), lagged causal arrows originate at faded
 * t−1 copies, contemporaneous arrows remain within t, and self-dynamics use
 * separate t−1 → matching-t edges. The same ELK spacing
 * (`DAG_LAYOUT_OPTIONS`) is used — so nodes land in the same columns. The
 * structural DAG leaves each glyph slot empty (drawing the edge straight through)
 * and uses compact cards instead of analysis's trajectory cards. It does not
 * cone-restrict: the structural stages show the full proposed model.
 */
export function StructureDag({
  constructs,
  edges,
  indicators,
  knownInputs = [],
  nodeStatuses,
  onNodeClick,
  direction = "RIGHT",
}: StructureDagProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const [hoverEdgeId, setHoverEdgeId] = useState<string | null>(null);
  const [dir, setDir] = useState<DagDirection>(direction);
  const [zoom, setZoom] = useState(0.69);
  const setZoomClamped = (z: number) => setZoom(Math.max(ZMIN, Math.min(ZMAX, z)));

  const byName = useMemo(() => new Map(constructs.map((c) => [c.name, c])), [constructs]);
  const knownInputByName = useMemo(
    () => new Map(knownInputs.map((input) => [input.construct, input])),
    [knownInputs],
  );

  const indicatorsByConstruct = useMemo(() => {
    const map = new Map<string, Indicator[]>();
    for (const ind of indicators ?? []) {
      const list = map.get(ind.construct_name);
      if (list) list.push(ind);
      else map.set(ind.construct_name, [ind]);
    }
    return map;
  }, [indicators]);

  const nodeWidth = indicatorsByConstruct.size > 0 ? NODE_W_WITH_INDICATORS : NODE_W;

  // Lagged cross-construct edges start at the cause's t−1 copy, while
  // contemporaneous edges stay within t. Endogenous time-varying self-dynamics
  // are added separately. The spacer preserves the established layout rhythm.
  const { graph, glyphs } = useMemo(() => {
    const timeVaryingNames = new Set(
      constructs
        .filter((construct) => construct.temporal_status === "time_varying")
        .map((construct) => construct.name),
    );
    const causalLinks = unrollCausalLinks(
      edges.filter((edge) => edge.cause !== edge.effect),
      timeVaryingNames,
    );
    const selfLinks = buildGhostLinks(
      constructs
        .filter(
          (construct) =>
            construct.role === "endogenous" && construct.temporal_status === "time_varying",
        )
        .map((construct) => ({ from: construct.name, to: construct.name })),
    );
    const ghosts = new Set([...causalLinks.ghosts, ...selfLinks.ghosts]);
    const pairs: GlyphPair[] = [
      ...causalLinks.edges.map((edge) => ({
        a: edge.source,
        b: edge.target,
        isSelf: false,
        lagged: edge.lagged,
      })),
      ...selfLinks.edges.map((edge) => ({
        a: edge.source,
        b: edge.target,
        isSelf: true,
        lagged: true,
      })),
    ];
    const split = splitEdgesWithGlyphs(pairs);
    const built: DagGraphInput = {
      nodes: [
        ...constructs.map((c) => ({
          id: c.name,
          width: nodeWidth,
          height: nodeHeight(indicatorsByConstruct.get(c.name)?.length ?? 0),
        })),
        ...[...ghosts].map((ghost) => ({ id: ghost, width: nodeWidth, height: HEADER_H })),
        ...split.glyphNodes,
      ],
      edges: split.edges,
      direction: dir,
      layoutOptions: DAG_LAYOUT_OPTIONS,
    };
    return { graph: built, glyphs: split.glyphs };
  }, [constructs, edges, indicatorsByConstruct, nodeWidth, dir]);

  const { nodes, edges: routed, width: W, height: H, isLayouting } = useDagLayout(graph);

  // Routed segment by id, to stitch each edge's `e<i>s` + `e<i>t` halves back together.
  const routedById = useMemo(() => new Map(routed.map((e) => [e.id, e])), [routed]);

  // The clicked construct plus its direct neighbors — everything else dims.
  const connected = useMemo(() => {
    if (!selected) return null;
    const set = new Set<string>([selected]);
    for (const e of edges) {
      if (e.cause === selected) set.add(e.effect);
      if (e.effect === selected) set.add(e.cause);
    }
    return set;
  }, [selected, edges]);

  const hoverEndpoints = useMemo(() => {
    const meta = hoverEdgeId ? glyphs.get(hoverEdgeId) : null;
    return meta ? new Set([baseId(meta.a), baseId(meta.b)]) : new Set<string>();
  }, [hoverEdgeId, glyphs]);

  const handleNodeClick = useCallback(
    (nodeId: string) => {
      const base = baseId(nodeId);
      setSelected((prev) => (prev === base ? null : base));
      onNodeClick?.(base);
    },
    [onNodeClick],
  );

  const hasGhosts = constructs.some(
    (construct) =>
      construct.temporal_status === "time_varying" &&
      (construct.role === "endogenous" ||
        edges.some((edge) => edge.lagged && edge.cause === construct.name)),
  );
  const hasCrossLagged = edges.some(
    (edge) => edge.lagged && byName.get(edge.cause)?.temporal_status === "time_varying",
  );
  const hasContemporaneous = edges.some((edge) => !edge.lagged);
  const statusValues = nodeStatuses ? Object.values(nodeStatuses) : [];
  const hasMarginalized = statusValues.includes("marginalized");
  const hasBlocking = statusValues.includes("blocking");

  return (
    <div style={{ fontFamily: "ui-sans-serif, system-ui, sans-serif" }}>
      {/* toolbar — mirrors the analysis interactive DAG */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 16,
          alignItems: "center",
          marginBottom: 10,
        }}
      >
        <DagDirectionToggle
          direction={dir === "RIGHT" ? "horizontal" : "vertical"}
          onDirectionChange={(direction) => setDir(direction === "horizontal" ? "RIGHT" : "DOWN")}
        />
        <DagZoomControls zoom={zoom} onZoomChange={setZoomClamped} />
      </div>

      {/* scrollable, content-sized canvas */}
      <DagCanvasFrame>
        {isLayouting ? null : (
          <DagSvg
            contentWidth={W}
            contentHeight={H}
            zoom={zoom}
            role="img"
            aria-label="Causal structure graph"
            onClick={(e) => {
              if (e.target === e.currentTarget) setSelected(null);
            }}
          >
            {[...glyphs.entries()].map(([gid, meta]) => {
              const i = gid.slice(3); // "G__<i>"
              const head = routedById.get(`e${i}s`);
              const tail = routedById.get(`e${i}t`);
              if (!head || !tail) return null;
              const points = cleanGlyphPath([...head.points, ...tail.points]);
              const aBase = baseId(meta.a);
              const bBase = baseId(meta.b);
              const { color, flagged } = edgeStatusColor(
                nodeStatuses?.[aBase],
                nodeStatuses?.[bBase],
              );
              const incident = selected == null || aBase === selected || bBase === selected;
              const isHover = hoverEdgeId === gid;
              const dimmed = selected != null && !incident && !isHover;
              return (
                <DagEdge
                  key={gid}
                  points={points}
                  color={color}
                  width={flagged ? 2.6 : 2}
                  opacity={dimmed ? 0.12 : meta.isSelf ? SELF_EDGE_OPACITY : 0.95}
                  highlighted={isHover}
                  onHoverChange={(h) => setHoverEdgeId(h ? gid : null)}
                />
              );
            })}
            {nodes.map((nd) => {
              if (nd.id.startsWith("G__")) return null; // empty glyph slot
              const base = baseId(nd.id);
              const construct = byName.get(base);
              if (!construct) return null;
              const prev = isGhost(nd.id);
              const lit = selected === base || hoverEndpoints.has(base);
              const dimmed = selected != null && !(connected?.has(base) ?? false);
              const baseOpacity = prev ? GHOST_OPACITY : 1;
              return (
                <g
                  key={nd.id}
                  transform={`translate(${nd.x} ${nd.y})`}
                  opacity={dimmed && !lit ? baseOpacity * 0.35 : baseOpacity}
                  style={{ cursor: "pointer", transition: "opacity 150ms" }}
                  onClick={() => handleNodeClick(nd.id)}
                >
                  <StructureNode
                    width={nd.width}
                    height={nd.height}
                    construct={construct}
                    indicators={indicatorsByConstruct.get(base) ?? []}
                    knownInput={knownInputByName.get(base)}
                    status={nodeStatuses?.[base]}
                    lit={lit}
                    isPrev={prev}
                  />
                </g>
              );
            })}
          </DagSvg>
        )}
      </DagCanvasFrame>

      {/* legend */}
      {hasGhosts || hasMarginalized || hasBlocking ? (
        <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 px-1 text-xs text-muted-foreground">
          {hasGhosts ? (
            <span className="flex items-center gap-1.5">
              <LegendSwatch border="var(--foreground)" faded />
              t−1 construct
            </span>
          ) : null}
          {hasCrossLagged ? <span>Lagged: cause(t−1) → effect(t)</span> : null}
          {hasContemporaneous ? <span>Contemporaneous: cause(t) → effect(t)</span> : null}
          {hasGhosts ? <span>Self-dynamics: construct(t−1) → construct(t)</span> : null}
          {hasMarginalized ? (
            <span className="flex items-center gap-1.5">
              <LegendSwatch border="var(--warning)" />
              marginalized
            </span>
          ) : null}
          {hasBlocking ? (
            <span className="flex items-center gap-1.5">
              <LegendSwatch border="var(--destructive)" />
              blocking
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
