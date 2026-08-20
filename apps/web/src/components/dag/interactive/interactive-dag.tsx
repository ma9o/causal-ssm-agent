"use client";

import type { CausalEdge, Construct, Indicator, KnownInput } from "@nof1-causal-lab/api-types";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import { DagCanvasFrame, DagSvg } from "../core/dag-canvas";
import { DagDirectionToggle } from "../core/dag-direction-toggle";
import { DagZoomControls } from "../core/dag-zoom-controls";
import { orthoPath } from "../core/ortho-path";
import { DAG_COLORS, signColor } from "../core/palette";
import {
  getEffectTrajectoryDays,
  getNodeActionSeries,
  getNodeReferenceSeries,
} from "../intervention-dag-semantics";
import type { AnalysisSimulationResult, EdgePosterior } from "../intervention-dag-types";
import type { ConstructStatus } from "../structure-dag";
import { baseId, buildSimulationGraph, CARD_H, CARD_W } from "./build-cone-graph";
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
const ZMIN = 0.4;
const ZMAX = 2.5;
const markerFor = (col: string) => (col === TEAL ? "arrPos" : col === RED ? "arrNeg" : "arrZero");

interface InteractiveDagProps {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  knownInputs?: KnownInput[];
  edgePosteriors?: Record<string, EdgePosterior>;
  persistencePosteriors?: Record<string, EdgePosterior>;
  identifiableTreatments?: string[];
  result: AnalysisSimulationResult;
  height?: number;
  onSimulate?: SimulateFn;
  /** Controlled indicator visibility. Omit to retain the DAG's local toggle. */
  indicatorsVisible?: boolean;
  /** Identification status keeps marginalized theory nodes visible in the fitted graph. */
  nodeStatuses?: Record<string, ConstructStatus>;
  onNodeClick?: (constructName: string) => void;
}

/**
 * Layer the backend's fitted and simulated artifacts over the full scientific
 * DAG. Theory-only structure remains visible; posterior summaries style fitted
 * edges; simulation results materialize reference and action node trajectories.
 */
export function InteractiveDag({
  constructs,
  edges,
  indicators = [],
  knownInputs = [],
  edgePosteriors = {},
  persistencePosteriors = {},
  identifiableTreatments = [],
  result,
  onSimulate,
  indicatorsVisible,
  nodeStatuses,
  onNodeClick,
}: InteractiveDagProps) {
  const outcome = result.outcome;
  const [dir, setDir] = useState<"LR" | "TB">("LR");
  const [localShowIndicators, setLocalShowIndicators] = useState(false);
  const showIndicators = indicatorsVisible ?? localShowIndicators;
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
  const { graph, edgeMeta } = useMemo(
    () =>
      buildSimulationGraph(constructs, edges, {
        dir: dir === "LR" ? "RIGHT" : "DOWN",
        showIndicators,
        showUnroll: true,
        indicators,
        persistenceNodes: Object.keys(persistencePosteriors),
      }),
    [constructs, edges, dir, showIndicators, indicators, persistencePosteriors],
  );
  const { nodes, edges: routed, width: W, height: H, isLayouting } = useDagLayout(graph);

  const identifiableTreatmentSet = useMemo(
    () => new Set(identifiableTreatments),
    [identifiableTreatments],
  );
  const knownInputSet = useMemo(
    () => new Set(knownInputs.map((input) => input.construct)),
    [knownInputs],
  );

  // Active interventions are read directly from the current backend result.
  const interventions = currentResult.clamps;
  const maximumPosteriorMean = useMemo(
    () =>
      Math.max(
        0,
        ...Object.values(edgePosteriors).map(({ mean }) => Math.abs(mean)),
        ...Object.values(persistencePosteriors).map(({ mean }) => Math.abs(mean)),
      ),
    [edgePosteriors, persistencePosteriors],
  );

  const setDo = useCallback(
    async (node: string, value: number) => {
      const fromDay = days[clampedDay];
      const horizonDay = days[n - 1];
      if (!onSimulate || fromDay == null || horizonDay == null) return;
      const horizonDays = Math.max(horizonDay, 1);
      const res = await onSimulate(
        buildSimulateInput(
          result,
          [{ variable: node, mode: "set", value, from_day: fromDay }],
          horizonDays,
        ),
      );
      setCurrentResult(res);
    },
    [onSimulate, days, clampedDay, n, result],
  );
  const resetScenario = useCallback(() => setCurrentResult(result), [result]);

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
        <DagDirectionToggle
          direction={dir === "LR" ? "horizontal" : "vertical"}
          onDirectionChange={(direction) => setDir(direction === "horizontal" ? "LR" : "TB")}
        />
        {indicatorsVisible === undefined ? (
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
              onChange={(e) => setLocalShowIndicators(e.target.checked)}
            />{" "}
            Indicators
          </label>
        ) : null}
        <DagZoomControls zoom={zoom} onZoomChange={setZoomClamped} />
      </div>

      {/* output */}
      <DagCanvasFrame>
        {isLayouting ? null : (
          <DagSvg contentWidth={W} contentHeight={H} zoom={zoom}>
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
              const meta = edgeMeta.get(e.id);
              if (!meta) return null;
              const { a, b, isSelf, lagged } = meta;
              const posterior = isSelf
                ? persistencePosteriors[baseId(b)]
                : edgePosteriors[`${baseId(a)}→${baseId(b)}`];
              const theoryOnly = !isSelf && posterior == null;
              const col = posterior ? signColor(posterior.mean) : NEUTRAL;
              const sourceStatus = nodeStatuses?.[baseId(a)];
              const targetStatus = nodeStatuses?.[baseId(b)];
              const contextOnly =
                sourceStatus === "marginalized" || targetStatus === "marginalized";
              const currentDay = days[clampedDay];
              const pruned =
                currentDay != null &&
                interventions.some(
                  (clamp) =>
                    clamp.variable === baseId(b) &&
                    clamp.from_day <= currentDay &&
                    (clamp.to_day == null || currentDay < clamp.to_day),
                );
              const key = `${a}>${b}`;
              const hl = hoverEdge === key;
              const d = orthoPath(e.points);
              const pts = e.points;
              const ep = pts[pts.length - 1];
              const fittedWidth = posterior
                ? 1.4 +
                  (maximumPosteriorMean > 0
                    ? (Math.abs(posterior.mean) / maximumPosteriorMean) * 3.4
                    : 0)
                : 1.2;
              return (
                <g key={e.id}>
                  <path
                    d={d}
                    fill="none"
                    stroke={contextOnly ? MUTED : pruned ? DAG_COLORS.pruned : col}
                    strokeWidth={pruned ? 1.4 : hl ? fittedWidth + 1.4 : fittedWidth}
                    strokeOpacity={contextOnly ? 0.3 : pruned ? 0.5 : theoryOnly ? 0.35 : 0.9}
                    strokeDasharray={
                      contextOnly || pruned || theoryOnly || lagged ? "5,4" : undefined
                    }
                    markerEnd={
                      !pruned ? `url(#${markerFor(contextOnly ? NEUTRAL : col)})` : undefined
                    }
                    style={hl ? { filter: "drop-shadow(0 0 2px rgba(20,25,30,.28))" } : undefined}
                  />
                  {pruned && ep ? (
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

            {/* node cards */}
            {nodes.map((nd) => {
              const base = baseId(nd.id);
              const isPrev = nd.id !== base;
              const construct = byName.get(base);
              if (!construct) return null;
              const referenceSeries = getNodeReferenceSeries(currentResult, base) ?? [];
              const actionSeries = getNodeActionSeries(currentResult, base) ?? [];
              const nodeInterventions = isPrev
                ? []
                : interventions.filter((clamp) => clamp.variable === base);
              const cardHl = hoverEndpoints.includes(base);
              const status = nodeStatuses?.[base];
              const contextOnly = status === "marginalized";
              return (
                <g
                  key={nd.id}
                  transform={`translate(${nd.x},${nd.y})`}
                  role={!isPrev && onNodeClick ? "button" : undefined}
                  tabIndex={!isPrev && onNodeClick ? 0 : undefined}
                  style={{
                    cursor: !isPrev && onNodeClick ? "pointer" : undefined,
                    ...(cardHl ? { filter: "drop-shadow(0 0 5px rgba(20,25,30,.22))" } : {}),
                  }}
                  onClick={() => {
                    if (!isPrev) onNodeClick?.(base);
                  }}
                  onKeyDown={(event) => {
                    if (!isPrev && (event.key === "Enter" || event.key === " ")) {
                      event.preventDefault();
                      onNodeClick?.(base);
                    }
                  }}
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
                    reference={referenceSeries}
                    action={actionSeries}
                    timeIndex={clampedDay}
                    interventions={nodeInterventions}
                    status={status}
                    knownInput={knownInputSet.has(base)}
                    persistence={persistencePosteriors[base]}
                    interactive={
                      !!onSimulate &&
                      n > 0 &&
                      referenceSeries.length === n &&
                      !isPrev &&
                      !contextOnly &&
                      identifiableTreatmentSet.has(base)
                    }
                    onSetDo={(v) => void setDo(base, v)}
                    onRemoveDo={currentResult !== result ? resetScenario : undefined}
                  />
                  {showIndicators && !isPrev ? (
                    <IndicatorStack
                      indicators={indicators.filter(
                        (indicator) => indicator.construct_name === base,
                      )}
                    />
                  ) : null}
                </g>
              );
            })}
          </DagSvg>
        )}
      </DagCanvasFrame>

      {/* graph note */}
      <div style={{ fontSize: 11.5, color: MUTED, margin: "8px 4px 0" }}>
        {`Showing the full scientific DAG for ${outcome} (${constructs.length} nodes); ★ marks the outcome.`}
        {Object.values(nodeStatuses ?? {}).includes("marginalized")
          ? "  ·  Marginalized confounders remain visible as subdued causal context."
          : null}
        {
          "  ·  Solid colored edges have fitted posterior coefficients; subdued dashed edges are theory-only context."
        }
      </div>

      {/* scrubber */}
      {n > 0 ? (
        <>
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
                max={Math.max(0, n - 1)}
                step={1}
                value={clampedDay}
                onChange={(e) => setDay(Number(e.target.value))}
                style={{ width: "100%", accentColor: INK }}
              />
              <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                {interventions.map((iv, index) => (
                  <span
                    key={`${iv.variable}-${iv.from_day}-${index}`}
                    style={{
                      position: "absolute",
                      top: 0,
                      bottom: 0,
                      left: `${(iv.from_day / Math.max(days[n - 1], 1)) * 100}%`,
                    }}
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
                        {`do · ${iv.variable.replace(/_/g, " ")} @d${iv.from_day}`}
                      </b>
                      {currentResult !== result ? (
                        <span
                          title="Reset to the selected scenario"
                          onClick={resetScenario}
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
                          ↺
                        </span>
                      ) : null}
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
              {`day ${days[clampedDay]}`}
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
            <span>{`${days[0]}d`}</span>
            <span>{`${days[Math.floor((n - 1) / 2)]}d`}</span>
            <span>{`${days[n - 1]}d`}</span>
          </div>
        </>
      ) : (
        <div
          style={{
            marginTop: 12,
            background: "#fff",
            border: `1px solid ${DAG_COLORS.line}`,
            borderRadius: 12,
            padding: "12px 16px",
            color: MUTED,
            fontSize: 11.5,
          }}
        >
          {`End-state result · effect ${currentResult.summary.mean.toFixed(3)} [${currentResult.summary.lower_95.toFixed(3)}, ${currentResult.summary.upper_95.toFixed(3)}] · no trajectory projection requested.`}
        </div>
      )}
    </div>
  );
}

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
