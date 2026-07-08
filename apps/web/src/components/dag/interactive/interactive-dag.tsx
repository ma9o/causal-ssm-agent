"use client";

import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import type { CausalEdge, Construct, Indicator } from "@nof1-causal-lab/api-types";
import { useCallback, useEffect, useMemo, useState } from "react";
import { orthoPath } from "../core/ortho-path";
import { clamp01, DAG_COLORS, signColor } from "../core/palette";
import {
  getEffectTrajectoryDays,
  getNodeActionSeries,
  getNodeReferenceSeries,
} from "../intervention-dag-semantics";
import type { AnalysisSimulationResult } from "../intervention-dag-types";
import { baseId, buildGlyphGraph, CARD_H, CARD_W, GLYPH_H, GLYPH_W } from "./build-cone-graph";
import {
  getAllEdgeDrift,
  getAllSelfEffects,
  getEdgeDrift,
  getNodeIndicators,
  getNodeRealized,
  getSelfEffect,
} from "./contract-extension";
import { DriftGlyph } from "./drift-glyph";
import { IndicatorStack } from "./indicator-stack";
import { buildSimulateInput, type SimulateFn } from "./simulate-input";
import { TrajectoryCard } from "./trajectory-card";

const {
  positive: TEAL,
  negative: RED,
  neutral: NEUTRAL,
  muted: MUTED,
  intervention: BLUE,
  ink: INK,
} = DAG_COLORS;
const TAG: Record<string, string> = { linear: "lin", hill: "Hill", mult: "×" };
const W_MIN = 1.0;
const W_MAX = 5.5;
const ZMIN = 0.4;
const ZMAX = 2.5;
const markerFor = (col: string) => (col === TEAL ? "arrPos" : col === RED ? "arrNeg" : "arrZero");

interface InteractiveDagProps {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  result: AnalysisSimulationResult;
  height?: number;
  onSimulate?: SimulateFn;
}

/**
 * analysis "Living DAG" — a faithful port of the design playground. Layered
 * cause→effect layout over the full projected estimation graph; every construct
 * carries its own counterfactual trajectory card; every cross-edge and
 * self-effect carries a two-panel drift glyph; one playhead sweeps them all;
 * per-node do() editing re-simulates via `onSimulate`.
 */
export function InteractiveDag({
  constructs,
  edges,
  indicators = [],
  result,
  onSimulate,
}: InteractiveDagProps) {
  const outcome = result.outcome;
  const [dir, setDir] = useState<"LR" | "TB">("LR");
  const [showIndicators, setShowIndicators] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [hoverEdge, setHoverEdge] = useState<string | null>(null);

  // Derive-from-props: a new scenario (result) resets in-progress do() editing.
  const [prevResult, setPrevResult] = useState(result);
  const [currentResult, setCurrentResult] = useState(result);
  if (prevResult !== result) {
    setPrevResult(result);
    setCurrentResult(result);
  }

  const days = useMemo(() => getEffectTrajectoryDays(currentResult), [currentResult]);
  const n = days.length;
  const [day, setDay] = useState(12);
  const [playing, setPlaying] = useState(false);
  const clampedDay = Math.max(0, Math.min(n - 1, day));
  useEffect(() => {
    if (!playing || n <= 1) return;
    const id = setInterval(() => setDay((d) => (d >= n - 1 ? 0 : d + 1)), 110);
    return () => clearInterval(id);
  }, [playing, n]);

  const byName = useMemo(() => new Map(constructs.map((c) => [c.name, c])), [constructs]);
  const { graph, glyphs } = useMemo(
    () =>
      buildGlyphGraph(constructs, edges, {
        dir: dir === "LR" ? "RIGHT" : "DOWN",
        showIndicators,
        showUnroll: true,
        indicators,
      }),
    [constructs, edges, dir, showIndicators, indicators],
  );
  const { nodes, edges: routed, width: W, height: H, isLayouting } = useDagLayout(graph);

  // active interventions, derived from the current result's clamps
  const interventions = useMemo(() => {
    const m = new Map<string, { day: number; value: number }>();
    for (const c of currentResult.clamps) {
      m.set(c.variable, { day: c.from_day ?? 0, value: c.value ?? c.amount ?? 0 });
    }
    return m;
  }, [currentResult]);

  // drift lookups + the global edge-width scale
  const contributionOf = useCallback(
    (a: string, b: string, isSelf: boolean): number[] =>
      isSelf
        ? (getSelfEffect(currentResult, baseId(b))?.contribution ?? [])
        : (getEdgeDrift(currentResult, a, b)?.contribution ?? []),
    [currentResult],
  );
  const absMax = useMemo(() => {
    let m = 0;
    for (const d of getAllEdgeDrift(currentResult))
      for (const c of d.contribution) m = Math.max(m, Math.abs(c));
    for (const s of getAllSelfEffects(currentResult))
      for (const c of s.contribution) m = Math.max(m, Math.abs(c));
    return m;
  }, [currentResult]);
  const widthOf = (c: number) =>
    absMax > 0
      ? Math.max(W_MIN, Math.min(W_MAX, W_MIN + (Math.abs(c) / absMax) * (W_MAX - W_MIN)))
      : 2;

  const movedOf = useCallback(
    (node: string) => {
      const ref = getNodeReferenceSeries(currentResult, node) ?? [];
      const act = getNodeActionSeries(currentResult, node) ?? [];
      return act.some((v, t) => Math.abs(v - (ref[t] ?? 0)) > 0.003);
    },
    [currentResult],
  );

  const setDo = useCallback(
    async (node: string, value: number) => {
      if (!onSimulate) return;
      const fromDay = days[clampedDay] ?? 0;
      const res = await onSimulate(
        buildSimulateInput(
          result,
          [{ variable: node, mode: "set", value: clamp01(value), from_day: fromDay }],
          60,
        ),
      );
      setCurrentResult(res);
    },
    [onSimulate, days, clampedDay, result],
  );
  const removeDo = useCallback(async () => {
    if (!onSimulate) {
      setCurrentResult(result);
      return;
    }
    setCurrentResult(await onSimulate(buildSimulateInput(result, [], 60)));
  }, [onSimulate, result]);

  const setZoomClamped = (z: number) => setZoom(Math.max(ZMIN, Math.min(ZMAX, z)));

  // column bands — shade alternate real-node layers
  const columnBands = useMemo(() => {
    const reals = nodes.filter((nd) => !nd.id.startsWith("G__"));
    const cols = new Map<number, { min: number; max: number }>();
    for (const nd of reals) {
      const key = Math.round(dir === "LR" ? nd.x : nd.y);
      const span =
        dir === "LR" ? { lo: nd.x, hi: nd.x + nd.width } : { lo: nd.y, hi: nd.y + nd.height };
      const cur = cols.get(key);
      if (cur) {
        cur.min = Math.min(cur.min, span.lo);
        cur.max = Math.max(cur.max, span.hi);
      } else cols.set(key, { min: span.lo, max: span.hi });
    }
    return [...cols.entries()]
      .sort((p, q) => p[0] - q[0])
      .map(([, v]) => v)
      .filter((_, i) => i % 2 === 1);
  }, [nodes, dir]);

  const hoverEndpoints = hoverEdge ? hoverEdge.split(">").map(baseId) : [];

  return (
    <div style={{ fontFamily: "ui-sans-serif, system-ui, sans-serif", color: INK, fontSize: 14 }}>
      {/* toolbar */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 16,
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <span style={LABEL}>Flow</span>
        <div
          style={{ display: "inline-flex", background: "#eef0f3", borderRadius: 10, padding: 3 }}
        >
          {(["LR", "TB"] as const).map((d) => (
            <button key={d} type="button" onClick={() => setDir(d)} style={segBtn(dir === d)}>
              {d === "LR" ? "→ left to right" : "↓ top to bottom"}
            </button>
          ))}
        </div>
        <label
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 7,
            fontSize: 13,
            color: "#4a4f57",
            cursor: "pointer",
            userSelect: "none",
          }}
        >
          <input
            type="checkbox"
            checked={showIndicators}
            onChange={(e) => setShowIndicators(e.target.checked)}
          />{" "}
          Indicators
        </label>
        <span style={LABEL}>Zoom</span>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <button
            type="button"
            onClick={() => setZoomClamped(zoom / 1.2)}
            style={zoomBtn}
            title="zoom out"
          >
            −
          </button>
          <span
            style={{
              fontVariantNumeric: "tabular-nums",
              fontSize: 12,
              color: "#4a4f57",
              minWidth: 40,
              textAlign: "center",
            }}
          >
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setZoomClamped(zoom * 1.2)}
            style={zoomBtn}
            title="zoom in"
          >
            +
          </button>
          <button type="button" onClick={() => setZoom(1)} style={zoomBtn} title="reset zoom">
            ⤢
          </button>
        </div>
      </div>

      {/* output */}
      <div
        style={{
          background: "#fff",
          border: `1px solid ${DAG_COLORS.line}`,
          borderRadius: 14,
          padding: 6,
          minHeight: 560,
          maxHeight: "74vh",
          overflow: "auto",
          backgroundImage: `radial-gradient(${DAG_COLORS.line} .8px, transparent .8px)`,
          backgroundSize: "18px 18px",
        }}
      >
        {isLayouting ? null : (
          <svg
            width={Math.ceil(W * zoom)}
            height={Math.ceil(H * zoom)}
            viewBox={`0 0 ${Math.ceil(W)} ${Math.ceil(H)}`}
            style={{ display: "block" }}
          >
            <defs>
              {(
                [
                  ["arrPos", TEAL],
                  ["arrNeg", RED],
                  ["arrZero", NEUTRAL],
                ] as const
              ).map(([id, col]) => (
                <marker
                  key={id}
                  id={id}
                  viewBox="0 0 10 10"
                  refX={9}
                  refY={5}
                  markerUnits="userSpaceOnUse"
                  markerWidth={11}
                  markerHeight={11}
                  orient="auto-start-reverse"
                >
                  <path d="M0,1 L9,5 L0,9" fill="none" stroke={col} strokeWidth={1.4} />
                </marker>
              ))}
            </defs>

            {columnBands.map((b, i) =>
              dir === "LR" ? (
                <rect
                  key={i}
                  x={b.min - 10}
                  y={0}
                  width={b.max - b.min + 20}
                  height={Math.ceil(H)}
                  fill={DAG_COLORS.col}
                />
              ) : (
                <rect
                  key={i}
                  x={0}
                  y={b.min - 10}
                  width={Math.ceil(W)}
                  height={b.max - b.min + 20}
                  fill={DAG_COLORS.col}
                />
              ),
            )}

            {/* edges */}
            {routed.map((e) => {
              const glyphId = e.source.startsWith("G__") ? e.source : e.target;
              const meta = glyphs.get(glyphId);
              if (!meta) return null;
              const { a, b, isSelf } = meta;
              if (clampedDay === 0 && a.endsWith("__p")) return null;
              const contrib = contributionOf(a, b, isSelf)[clampedDay] ?? 0;
              const col = signColor(contrib);
              const tiv = interventions.get(b);
              const pruned = !!tiv && tiv.day === days[clampedDay];
              const head = !e.target.startsWith("G__");
              const key = `${a}>${b}`;
              const hl = hoverEdge === key;
              const d = orthoPath(e.points);
              const pts = e.points;
              const ep = pts[pts.length - 1];
              return (
                <g key={e.id}>
                  <path
                    d={d}
                    fill="none"
                    stroke={pruned ? DAG_COLORS.pruned : col}
                    strokeWidth={pruned ? 1.4 : hl ? 4.5 : widthOf(contrib)}
                    strokeOpacity={pruned ? 0.5 : 1}
                    strokeDasharray={pruned ? "5,4" : undefined}
                    markerEnd={head && !pruned ? `url(#${markerFor(col)})` : undefined}
                    style={hl ? { filter: "drop-shadow(0 0 2px rgba(20,25,30,.28))" } : undefined}
                  />
                  {pruned && head && ep ? (
                    <text
                      x={ep.x - 9}
                      y={ep.y - 4}
                      textAnchor="middle"
                      fontSize={10}
                      fill={DAG_COLORS.scissors}
                    >
                      ✂
                    </text>
                  ) : null}
                  <path
                    d={d}
                    fill="none"
                    stroke="transparent"
                    strokeWidth={16}
                    pointerEvents="stroke"
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHoverEdge(key)}
                    onMouseLeave={() => setHoverEdge(null)}
                  />
                </g>
              );
            })}

            {/* glyph nodes */}
            {nodes.map((nd) => {
              if (!nd.id.startsWith("G__")) return null;
              const meta = glyphs.get(nd.id);
              if (!meta) return null;
              const { a, b, isSelf } = meta;
              if (clampedDay === 0 && a.endsWith("__p")) return null;
              const tiv = interventions.get(b);
              const pruned = !!tiv && tiv.day === days[clampedDay];
              const drift = isSelf ? null : getEdgeDrift(currentResult, a, b);
              const self = isSelf ? getSelfEffect(currentResult, baseId(b)) : null;
              const transfer = isSelf ? (self?.transfer ?? []) : (drift?.transfer ?? []);
              const contribution = isSelf
                ? (self?.contribution ?? [])
                : (drift?.contribution ?? []);
              const driverLevel = isSelf ? (self?.level ?? []) : (drift?.driver_level ?? []);
              const col = pruned ? NEUTRAL : signColor(contribution[clampedDay] ?? 0);
              const key = `${a}>${b}`;
              return (
                <g
                  key={nd.id}
                  transform={`translate(${nd.x},${nd.y})`}
                  opacity={pruned ? 0.3 : 1}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHoverEdge(key)}
                  onMouseLeave={() => setHoverEdge(null)}
                >
                  <DriftGlyph
                    width={GLYPH_W}
                    height={GLYPH_H}
                    transfer={transfer}
                    contribution={contribution}
                    driverLevel={driverLevel}
                    timeIndex={clampedDay}
                    color={col}
                    label={isSelf ? "self" : TAG[drift?.form ?? "linear"]}
                    xlabel={isSelf ? "vs level" : "vs cause"}
                    highlighted={hoverEdge === key}
                  />
                </g>
              );
            })}

            {/* node cards */}
            {nodes.map((nd) => {
              if (nd.id.startsWith("G__")) return null;
              const base = baseId(nd.id);
              const isPrev = nd.id !== base;
              if (clampedDay === 0 && isPrev) return null;
              const construct = byName.get(base);
              if (!construct) return null;
              const iv = isPrev ? null : (interventions.get(base) ?? null);
              const cardHl = hoverEndpoints.includes(base);
              return (
                <g
                  key={nd.id}
                  transform={`translate(${nd.x},${nd.y})`}
                  style={cardHl ? { filter: "drop-shadow(0 0 5px rgba(20,25,30,.22))" } : undefined}
                >
                  <TrajectoryCard
                    width={CARD_W}
                    height={CARD_H}
                    name={base}
                    kind={construct.role === "endogenous" ? "endo" : "exo"}
                    vary={construct.temporal_status === "time_varying" ? "varying" : "invariant"}
                    isTarget={construct.is_outcome}
                    isPrev={isPrev}
                    days={days}
                    reference={getNodeReferenceSeries(currentResult, base) ?? []}
                    action={getNodeActionSeries(currentResult, base) ?? []}
                    realized={getNodeRealized(currentResult, base)}
                    timeIndex={clampedDay}
                    intervention={iv}
                    interactive={!!onSimulate && !isPrev}
                    otherActive={interventions.size > 0 && !interventions.has(base)}
                    onSetDo={(v) => void setDo(base, v)}
                    onRemoveDo={() => void removeDo()}
                  />
                  {showIndicators && !isPrev ? (
                    <IndicatorStack
                      indicators={getNodeIndicators(currentResult, base)}
                      days={days}
                      timeIndex={clampedDay}
                      moved={movedOf(base)}
                    />
                  ) : null}
                </g>
              );
            })}
          </svg>
        )}
      </div>

      {/* graph note */}
      <div style={{ fontSize: 11.5, color: MUTED, margin: "8px 4px 0" }}>
        {`Showing the estimation graph for ${outcome} (${constructs.length} nodes); ★ marks the outcome.`}
        {
          "  ·  Unrolled: each latent's faded t−1 card feeds its present-time self (the self-edge = its NodePotential −dV/dη)."
        }
      </div>

      {/* scrubber */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 14,
          marginTop: 12,
          background: "#fff",
          border: `1px solid ${DAG_COLORS.line}`,
          borderRadius: 12,
          padding: "12px 16px",
        }}
      >
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          style={iconBtn}
          title="play / pause"
        >
          {playing ? "⏸" : "▶"}
        </button>
        <button
          type="button"
          onClick={() => {
            setPlaying(false);
            setDay(0);
          }}
          style={iconBtn}
          title="reset"
        >
          ↺
        </button>
        <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center" }}>
          <input
            type="range"
            min={0}
            max={60}
            step={1}
            value={clampedDay}
            onChange={(e) => setDay(Number(e.target.value))}
            style={{ width: "100%", accentColor: INK }}
          />
          <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
            {[...interventions.entries()].map(([id, iv]) => (
              <span
                key={id}
                style={{ position: "absolute", top: 0, bottom: 0, left: `${(iv.day / 60) * 100}%` }}
              >
                <i
                  style={{
                    position: "absolute",
                    top: 2,
                    bottom: 2,
                    left: -1.25,
                    width: 2.5,
                    background: BLUE,
                    borderRadius: 2,
                  }}
                />
                <span
                  style={{
                    position: "absolute",
                    top: -16,
                    left: 0,
                    transform: "translateX(-50%)",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 3,
                  }}
                >
                  <b
                    style={{
                      fontSize: 9,
                      fontWeight: 600,
                      color: "#fff",
                      background: BLUE,
                      borderRadius: 4,
                      padding: "1px 5px",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {`do · ${id.replace(/_/g, " ")} @d${iv.day}`}
                  </b>
                  <span
                    onClick={() => void removeDo()}
                    style={{
                      cursor: "pointer",
                      pointerEvents: "auto",
                      fontSize: 9,
                      fontWeight: 700,
                      color: "#fff",
                      background: RED,
                      borderRadius: 4,
                      padding: "1px 4px",
                    }}
                  >
                    ✕
                  </span>
                </span>
              </span>
            ))}
          </div>
        </div>
        <span
          style={{
            fontVariantNumeric: "tabular-nums",
            minWidth: 96,
            textAlign: "right",
            color: "#3a3f47",
          }}
        >
          {`day ${clampedDay}`}
        </span>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          color: MUTED,
          margin: "6px 8px 0",
        }}
      >
        <span>1d</span>
        <span>7d</span>
        <span>30d</span>
        <span>60d</span>
      </div>
    </div>
  );
}

const LABEL: React.CSSProperties = {
  fontSize: 11,
  letterSpacing: ".04em",
  textTransform: "uppercase",
  color: MUTED,
};
const zoomBtn: React.CSSProperties = {
  width: 26,
  height: 26,
  border: `1px solid ${DAG_COLORS.line2}`,
  background: "#fff",
  borderRadius: 7,
  cursor: "pointer",
  fontSize: 14,
  lineHeight: 1,
  display: "grid",
  placeItems: "center",
  color: "#4a4f57",
};
const iconBtn: React.CSSProperties = {
  border: `1px solid ${DAG_COLORS.line2}`,
  background: "#fff",
  width: 34,
  height: 34,
  borderRadius: 9,
  cursor: "pointer",
  fontSize: 15,
  display: "grid",
  placeItems: "center",
};
const segBtn = (on: boolean): React.CSSProperties => ({
  border: 0,
  background: on ? "#fff" : "transparent",
  padding: "7px 12px",
  borderRadius: 8,
  fontSize: 13,
  color: on ? INK : "#4a4f57",
  cursor: "pointer",
  fontWeight: on ? 600 : 400,
  boxShadow: on ? "0 1px 2px rgba(0,0,0,.08)" : undefined,
});
