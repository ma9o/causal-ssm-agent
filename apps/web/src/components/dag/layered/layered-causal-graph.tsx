"use client";

import type { Construct, Indicator, LikelihoodSpec } from "@nof1-causal-lab/api-types";
import { Pause, Play } from "lucide-react";
import { type KeyboardEvent, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import type { DagLayoutNode } from "@/lib/utils/dag-graph-layout";
import { deriveConstructStatuses } from "../construct-statuses";
import { DagCanvasFrame, DagSvg } from "../core/dag-canvas";
import { DagEdge } from "../core/dag-edge";
import { DagNodeShell } from "../core/dag-node";
import { DAG_COLORS, signColor } from "../core/palette";
import { DagZoomControls } from "../core/dag-zoom-controls";
import {
  getEffectTrajectoryDays,
  getNodeActionSeries,
  getNodeReferenceSeries,
} from "../intervention-dag-semantics";
import type { EdgePosterior } from "../intervention-dag-types";
import type { ConstructStatus } from "../structure-dag";
import {
  buildLayeredCausalGraph,
  LAYERED_EDGE_SLOT_HEIGHT,
  LAYERED_EDGE_SLOT_WIDTH,
  LAYERED_HISTORY_HEIGHT,
  LAYERED_HISTORY_WIDTH,
  LAYERED_NODE_HEIGHT,
  LAYERED_NODE_WIDTH,
  type LayeredGraphEdgeMeta,
} from "./build-layered-causal-graph";
import {
  availableGraphLayers,
  causalEdgeKey,
  type CausalGraphLayerId,
  deriveEdgeDesignDispositions,
  type EdgeDesignDisposition,
  type LayeredCausalGraphModel,
} from "./layered-causal-graph-model";

const CANVAS_PADDING = 36;
const MIN_ZOOM = 0.42;
const MAX_ZOOM = 1.8;
const BLOCKING = "#dc2626";
const MARGINALIZED = "#d97706";

const LAYER_LABELS: Record<CausalGraphLayerId, string> = {
  structure: "Structure",
  measurement: "Measurement",
  design: "Design",
  specification: "Specification",
  fit: "Fit",
  simulation: "Simulation",
};

export interface LayeredCausalGraphProps {
  model: LayeredCausalGraphModel;
  initialSelectedNode?: string | null;
}

function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

function truncate(value: string, length: number): string {
  return value.length > length ? `${value.slice(0, Math.max(1, length - 1))}…` : value;
}

function activateOnKeyboard(event: KeyboardEvent<SVGGElement>, action: () => void): void {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

function wrapDescription(value: string, lineLength = 40): [string, string] {
  const words = value.trim().split(/\s+/);
  let first = "";
  let index = 0;
  for (; index < words.length; index += 1) {
    const candidate = first ? `${first} ${words[index]}` : words[index];
    if (candidate.length > lineLength && first) break;
    first = candidate;
  }
  return [truncate(first, lineLength + 1), truncate(words.slice(index).join(" "), lineLength + 1)];
}

function statusAccent(status: ConstructStatus | undefined): string | undefined {
  if (status === "blocking") return BLOCKING;
  if (status === "marginalized") return MARGINALIZED;
  return undefined;
}

function statusLabel(status: ConstructStatus | undefined): string | null {
  if (status === "blocking") return "blocking";
  if (status === "marginalized") return "marginalized";
  return null;
}

function LayerPill({ x, label, color }: { x: number; label: string; color: string }) {
  const width = Math.max(38, label.length * 5.3 + 14);
  return (
    <g transform={`translate(${x - width},8)`}>
      <rect width={width} height={17} rx={8.5} fill={color} fillOpacity={0.11} />
      <text x={width / 2} y={11.5} textAnchor="middle" fontSize={7.5} fontWeight={650} fill={color}>
        {label}
      </text>
    </g>
  );
}

function pathForSeries(
  series: number[],
  x: number,
  y: number,
  width: number,
  height: number,
  minimum: number,
  maximum: number,
): string {
  const range = maximum - minimum || 1;
  const denominator = Math.max(1, series.length - 1);
  return series
    .map((value, index) => {
      const px = x + (index / denominator) * width;
      const py = y + height - ((value - minimum) / range) * height;
      return `${index === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join("");
}

function MiniTrajectory({
  days,
  reference,
  action,
  dayIndex,
}: {
  days: number[];
  reference: number[];
  action: number[];
  dayIndex: number;
}) {
  if (reference.length !== days.length) {
    throw new Error("A simulation reference trajectory is not aligned to its day axis.");
  }
  if (action.length > 0 && action.length !== days.length) {
    throw new Error("A simulation action trajectory is not aligned to its day axis.");
  }
  if (reference.length === 0) return null;

  const values = [...reference, ...action];
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const x = 14;
  const y = 77;
  const width = LAYERED_NODE_WIDTH - 28;
  const height = 27;
  const index = Math.max(0, Math.min(reference.length - 1, dayIndex));
  const markerX = x + (index / Math.max(1, reference.length - 1)) * width;

  return (
    <g>
      <line x1={x} x2={x + width} y1={y + height / 2} y2={y + height / 2} stroke="#edf0f3" />
      <path
        d={pathForSeries(reference, x, y, width, height, minimum, maximum)}
        fill="none"
        stroke={DAG_COLORS.slate}
        strokeWidth={1.35}
        strokeOpacity={0.7}
      />
      {action.length > 0 ? (
        <path
          d={pathForSeries(action, x, y, width, height, minimum, maximum)}
          fill="none"
          stroke={DAG_COLORS.intervention}
          strokeWidth={1.8}
        />
      ) : null}
      <line
        x1={markerX}
        x2={markerX}
        y1={y - 2}
        y2={y + height + 2}
        stroke={DAG_COLORS.ink}
        strokeOpacity={0.18}
      />
    </g>
  );
}

function measurementSummary(
  indicators: Indicator[],
  likelihoodByVariable: ReadonlyMap<string, LikelihoodSpec>,
  warningVariables: ReadonlySet<string>,
): string {
  const rendered = indicators.slice(0, 2).map((indicator) => {
    const likelihood = likelihoodByVariable.get(indicator.name);
    const suffix = likelihood ? `:${likelihood.distribution}` : `:${indicator.measurement_dtype}`;
    return `${warningVariables.has(indicator.name) ? "!" : "•"} ${truncate(humanize(indicator.name), 15)}${suffix}`;
  });
  if (indicators.length > 2) rendered.push(`+${indicators.length - 2}`);
  return rendered.join("  ");
}

function ConstructCard({
  construct,
  indicators,
  likelihoodByVariable,
  warningVariables,
  status,
  knownInput,
  persistence,
  days,
  reference,
  action,
  dayIndex,
  clampLabel,
  selected,
  dimmed,
  onSelect,
}: {
  construct: Construct;
  indicators: Indicator[];
  likelihoodByVariable: ReadonlyMap<string, LikelihoodSpec>;
  warningVariables: ReadonlySet<string>;
  status?: ConstructStatus;
  knownInput: boolean;
  persistence?: EdgePosterior;
  days: number[];
  reference: number[];
  action: number[];
  dayIndex: number;
  clampLabel?: string;
  selected: boolean;
  dimmed: boolean;
  onSelect: () => void;
}) {
  const [descriptionLine1, descriptionLine2] = wrapDescription(construct.description);
  const label = statusLabel(status);
  const accent = clampLabel ? DAG_COLORS.intervention : statusAccent(status);
  const summary = measurementSummary(indicators, likelihoodByVariable, warningVariables);
  const badge = clampLabel ?? label ?? (knownInput ? "known input" : null);
  const badgeColor = clampLabel
    ? DAG_COLORS.intervention
    : status === "blocking"
      ? BLOCKING
      : status === "marginalized"
        ? MARGINALIZED
        : DAG_COLORS.slate;

  return (
    <g
      opacity={dimmed ? 0.18 : status === "marginalized" ? 0.62 : 1}
      role="button"
      tabIndex={0}
      style={{ cursor: "pointer" }}
      onClick={onSelect}
      onKeyDown={(event) => activateOnKeyboard(event, onSelect)}
    >
      <DagNodeShell
        width={LAYERED_NODE_WIDTH}
        height={LAYERED_NODE_HEIGHT}
        title={`${construct.is_outcome ? "★ " : ""}${truncate(humanize(construct.name), 29)}`}
        subtitle={`${construct.role} · ${construct.temporal_status === "time_varying" ? "varying" : "invariant"}`}
        accent={selected ? "var(--primary)" : accent}
        dashed={status === "marginalized"}
        highlighted={selected}
        outcome={construct.is_outcome}
      >
        {badge ? <LayerPill x={LAYERED_NODE_WIDTH - 8} label={badge} color={badgeColor} /> : null}
        <text x={14} y={61} fontSize={8.2} fill="var(--muted-foreground)">
          {descriptionLine1}
        </text>
        <text x={14} y={72} fontSize={8.2} fill="var(--muted-foreground)">
          {descriptionLine2}
        </text>
        {days.length > 0 && reference.length > 0 ? (
          <MiniTrajectory days={days} reference={reference} action={action} dayIndex={dayIndex} />
        ) : persistence ? (
          <text
            x={14}
            y={94}
            fontSize={8}
            fontFamily="ui-monospace, monospace"
            fill={DAG_COLORS.muted}
          >
            {`ρ ${persistence.mean.toFixed(2)} [${persistence.ci_lower.toFixed(2)}, ${persistence.ci_upper.toFixed(2)}]`}
          </text>
        ) : null}
        {indicators.length > 0 ? (
          <g>
            <line x1={12} x2={LAYERED_NODE_WIDTH - 12} y1={110} y2={110} stroke="var(--border)" />
            <text
              x={14}
              y={125}
              fontSize={7.2}
              fontFamily="ui-monospace, monospace"
              fill={DAG_COLORS.muted}
            >
              {truncate(summary, 61)}
            </text>
          </g>
        ) : null}
      </DagNodeShell>
    </g>
  );
}

function HistoryCard({
  construct,
  status,
  persistence,
  dimmed,
  selected,
  onSelect,
}: {
  construct: Construct;
  status?: ConstructStatus;
  persistence?: EdgePosterior;
  dimmed: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <g
      opacity={dimmed ? 0.13 : 0.45}
      role="button"
      tabIndex={0}
      style={{ cursor: "pointer" }}
      onClick={onSelect}
      onKeyDown={(event) => activateOnKeyboard(event, onSelect)}
    >
      <DagNodeShell
        width={LAYERED_HISTORY_WIDTH}
        height={LAYERED_HISTORY_HEIGHT}
        title={`${truncate(humanize(construct.name), 20)} · t−1`}
        subtitle={
          persistence
            ? `fitted persistence ρ ${persistence.mean.toFixed(2)}`
            : "previous-time state"
        }
        accent={selected ? "var(--primary)" : statusAccent(status)}
        dashed
        highlighted={selected}
      />
    </g>
  );
}

function EdgeSlot({
  meta,
  disposition,
  posterior,
  color,
  pruned,
  specificationVisible,
  dimmed,
}: {
  meta: LayeredGraphEdgeMeta;
  disposition?: EdgeDesignDisposition;
  posterior?: EdgePosterior;
  color: string;
  pruned: boolean;
  specificationVisible: boolean;
  dimmed: boolean;
}) {
  const top = meta.isSelf ? "AR(1)" : meta.lagged ? "lag 1" : "same t";
  const bottom = pruned
    ? "cut by do()"
    : posterior
      ? `${posterior.mean >= 0 ? "+" : ""}${posterior.mean.toFixed(2)}`
      : disposition === "projected_edge"
        ? "projected"
        : specificationVisible
          ? meta.isSelf
            ? "ρ prior"
            : "β prior"
          : disposition === "retained_edge"
            ? "retained"
            : null;

  return (
    <g opacity={dimmed ? 0.12 : 1}>
      <rect
        width={LAYERED_EDGE_SLOT_WIDTH}
        height={LAYERED_EDGE_SLOT_HEIGHT}
        rx={8}
        fill="var(--card)"
        stroke={color}
        strokeOpacity={0.55}
        strokeDasharray={disposition === "projected_edge" ? "4,3" : undefined}
      />
      <text
        x={LAYERED_EDGE_SLOT_WIDTH / 2}
        y={bottom ? 12 : 19}
        textAnchor="middle"
        fontSize={7.2}
        fontWeight={650}
        fill={color}
      >
        {top}
      </text>
      {bottom ? (
        <text
          x={LAYERED_EDGE_SLOT_WIDTH / 2}
          y={24}
          textAnchor="middle"
          fontSize={7.2}
          fontFamily="ui-monospace, monospace"
          fill={color}
        >
          {bottom}
        </text>
      ) : null}
    </g>
  );
}

interface GraphBand {
  key: "static" | "history" | "present";
  label: string;
  nodes: DagLayoutNode[];
}

function boundsForBand(
  band: GraphBand,
): { x: number; y: number; width: number; height: number } | null {
  if (band.nodes.length === 0) return null;
  const minimumX = Math.min(...band.nodes.map((node) => node.x));
  const minimumY = Math.min(...band.nodes.map((node) => node.y));
  const maximumX = Math.max(...band.nodes.map((node) => node.x + node.width));
  const maximumY = Math.max(...band.nodes.map((node) => node.y + node.height));
  return {
    x: minimumX - 14,
    y: minimumY - 28,
    width: maximumX - minimumX + 28,
    height: maximumY - minimumY + 42,
  };
}

function LayerControls({
  available,
  visible,
  onToggle,
}: {
  available: CausalGraphLayerId[];
  visible: ReadonlySet<CausalGraphLayerId>;
  onToggle: (layer: CausalGraphLayerId) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5" aria-label="Causal graph layers">
      {available.map((layer, index) => {
        const active = visible.has(layer);
        return (
          <button
            key={layer}
            type="button"
            disabled={layer === "structure"}
            aria-pressed={active}
            onClick={() => onToggle(layer)}
            className={`rounded-full border px-2.5 py-1 text-[10px] font-medium transition-colors ${
              active
                ? "border-slate-500 bg-slate-800 text-white"
                : "border-slate-200 bg-white text-slate-400"
            } disabled:cursor-default disabled:opacity-100`}
          >
            {`${index + 1} · ${LAYER_LABELS[layer]}`}
          </button>
        );
      })}
    </div>
  );
}

export function LayeredCausalGraph({ model, initialSelectedNode = null }: LayeredCausalGraphProps) {
  const available = useMemo(() => availableGraphLayers(model), [model]);
  const [hiddenLayers, setHiddenLayers] = useState<Set<CausalGraphLayerId>>(() => new Set());
  const [selectedNode, setSelectedNode] = useState<string | null>(initialSelectedNode);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);
  const [zoom, setZoom] = useState(0.72);
  const [dayIndex, setDayIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const visible = useMemo(
    () => new Set(available.filter((layer) => !hiddenLayers.has(layer))),
    [available, hiddenLayers],
  );

  const topology = useMemo(() => buildLayeredCausalGraph(model.structure), [model.structure]);
  const { nodes, edges: routedSegments, width, height, isLayouting } = useDagLayout(topology.graph);

  const measurementLayer = visible.has("measurement") ? model.measurement : undefined;
  const designLayer = visible.has("design") ? model.design : undefined;
  const specificationLayer = visible.has("specification") ? model.specification : undefined;
  const fitLayer = visible.has("fit") ? model.fit : undefined;
  const simulationLayer = visible.has("simulation") ? model.simulation : undefined;
  const designVisible = designLayer != null;
  const specificationVisible = specificationLayer != null;
  const fitVisible = fitLayer != null;
  const simulationVisible = simulationLayer != null;

  const nodeStatuses = useMemo(
    () =>
      designLayer
        ? deriveConstructStatuses(designLayer.causalDesign, designLayer.structuralPlan)
        : {},
    [designLayer],
  );
  const edgeDispositions = useMemo(
    () =>
      designLayer
        ? deriveEdgeDesignDispositions(designLayer.structuralPlan)
        : new Map<string, EdgeDesignDisposition>(),
    [designLayer],
  );
  const indicatorsByConstruct = useMemo(() => {
    const result = new Map<string, Indicator[]>();
    if (!measurementLayer) return result;
    for (const indicator of measurementLayer.measurement.indicators) {
      const indicators = result.get(indicator.construct_name) ?? [];
      indicators.push(indicator);
      result.set(indicator.construct_name, indicators);
    }
    return result;
  }, [measurementLayer]);
  const knownInputNames = useMemo(
    () =>
      new Set(
        measurementLayer
          ? measurementLayer.knownInputs.map((knownInput) => knownInput.construct)
          : [],
      ),
    [measurementLayer],
  );
  const likelihoodByVariable = useMemo(
    () =>
      new Map(
        specificationLayer
          ? specificationLayer.modelSpec.statistical_model_spec.likelihoods.map(
              (likelihood) => [likelihood.variable, likelihood] as const,
            )
          : [],
      ),
    [specificationLayer],
  );
  const warningVariables = useMemo(
    () =>
      new Set(
        fitLayer
          ? fitLayer.posterior.ppc.per_variable_warnings
              .filter((warning) => !warning.passed)
              .map((warning) => warning.variable)
          : [],
      ),
    [fitLayer],
  );
  const edgePosteriors: Record<string, EdgePosterior> = fitLayer?.edgePosteriors ?? {};
  const persistencePosteriors: Record<string, EdgePosterior> =
    fitLayer?.persistencePosteriors ?? {};
  const maximumPosteriorMean = Math.max(
    0,
    ...Object.values(edgePosteriors).map((posterior) => Math.abs(posterior.mean)),
    ...Object.values(persistencePosteriors).map((posterior) => Math.abs(posterior.mean)),
  );

  const simulationResult = simulationLayer?.result ?? null;
  const days = useMemo(
    () => (simulationResult ? getEffectTrajectoryDays(simulationResult) : []),
    [simulationResult],
  );
  const clampedDayIndex = Math.max(0, Math.min(Math.max(0, days.length - 1), dayIndex));
  const currentDay = days[clampedDayIndex];
  useEffect(() => {
    if (!playing || days.length < 2) return;
    const timer = window.setInterval(
      () => setDayIndex((current) => (current >= days.length - 1 ? 0 : current + 1)),
      180,
    );
    return () => window.clearInterval(timer);
  }, [playing, days.length]);

  const selectedNeighborhood = useMemo(() => {
    if (!selectedNode) return null;
    const names = new Set([selectedNode]);
    for (const edge of model.structure.edges) {
      if (edge.cause === selectedNode) names.add(edge.effect);
      if (edge.effect === selectedNode) names.add(edge.cause);
    }
    return names;
  }, [model.structure.edges, selectedNode]);

  const graphBands = useMemo<GraphBand[]>(() => {
    const staticNodes: DagLayoutNode[] = [];
    const historyNodes: DagLayoutNode[] = [];
    const presentNodes: DagLayoutNode[] = [];
    for (const node of nodes) {
      const meta = topology.nodeMeta.get(node.id);
      if (meta?.kind === "history") historyNodes.push(node);
      if (meta?.kind === "construct") {
        if (meta.construct.temporal_status === "time_invariant") staticNodes.push(node);
        else presentNodes.push(node);
      }
    }
    return [
      { key: "static", label: "stable context", nodes: staticNodes },
      { key: "history", label: "t−1", nodes: historyNodes },
      { key: "present", label: "t", nodes: presentNodes },
    ];
  }, [nodes, topology.nodeMeta]);

  const toggleLayer = (layer: CausalGraphLayerId) => {
    if (layer === "structure") return;
    setHiddenLayers((current) => {
      const next = new Set(current);
      if (next.has(layer)) next.delete(layer);
      else next.add(layer);
      return next;
    });
  };

  const edgeVisual = (meta: LayeredGraphEdgeMeta) => {
    const disposition = meta.isSelf
      ? nodeStatuses[meta.cause] === "marginalized"
        ? "projected_edge"
        : undefined
      : edgeDispositions.get(causalEdgeKey(meta.cause, meta.effect, meta.lagged));
    const posterior = meta.isSelf
      ? persistencePosteriors[meta.cause]
      : edgePosteriors[`${meta.cause}→${meta.effect}`];
    const activeClamp =
      currentDay != null &&
      simulationResult?.clamps.some(
        (clamp) =>
          clamp.variable === meta.effect &&
          clamp.from_day <= currentDay &&
          (clamp.to_day == null || currentDay < clamp.to_day),
      );
    const blocking =
      nodeStatuses[meta.cause] === "blocking" || nodeStatuses[meta.effect] === "blocking";
    const marginalized =
      nodeStatuses[meta.cause] === "marginalized" || nodeStatuses[meta.effect] === "marginalized";
    const color = activeClamp
      ? DAG_COLORS.pruned
      : blocking
        ? BLOCKING
        : marginalized
          ? MARGINALIZED
          : posterior
            ? signColor(posterior.mean)
            : disposition === "projected_edge"
              ? DAG_COLORS.muted
              : meta.lagged
                ? DAG_COLORS.lagged
                : DAG_COLORS.contemporaneous;
    const width = posterior
      ? 1.5 +
        (maximumPosteriorMean > 0 ? (Math.abs(posterior.mean) / maximumPosteriorMean) * 3.5 : 0)
      : 1.7;
    const selectedEdge =
      selectedNode == null || meta.cause === selectedNode || meta.effect === selectedNode;
    const dimmed = !selectedEdge || (hoveredEdge != null && hoveredEdge !== meta.id);
    const opacity = dimmed
      ? 0.1
      : activeClamp
        ? 0.45
        : marginalized || disposition === "projected_edge"
          ? 0.38
          : posterior
            ? 0.92
            : 0.78;
    return {
      disposition,
      posterior,
      activeClamp: Boolean(activeClamp),
      color,
      width,
      opacity,
      dimmed,
    };
  };

  return (
    <div className="min-h-0 space-y-3 bg-slate-50/60 p-3">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-white px-3 py-2.5">
        <div>
          <div className="text-xs font-semibold text-slate-800">Causal model</div>
          <div className="text-[10px] text-muted-foreground">
            One structural topology; each materialized artifact adds one visual layer.
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <LayerControls available={available} visible={visible} onToggle={toggleLayer} />
          <DagZoomControls
            zoom={zoom}
            onZoomChange={(next) => setZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, next)))}
          />
        </div>
      </div>

      <DagCanvasFrame>
        {isLayouting ? (
          <div className="h-[560px] animate-pulse rounded-xl bg-slate-100" />
        ) : (
          <DagSvg
            contentWidth={width + CANVAS_PADDING * 2}
            contentHeight={height + CANVAS_PADDING * 2}
            zoom={zoom}
            role="img"
            aria-label="Layered causal graph"
          >
            <g transform={`translate(${CANVAS_PADDING},${CANVAS_PADDING})`}>
              {graphBands.map((band) => {
                const bounds = boundsForBand(band);
                if (!bounds) return null;
                return (
                  <g key={band.key}>
                    <rect
                      {...bounds}
                      rx={14}
                      fill={band.key === "present" ? "#f8fafc" : "#fbfcfd"}
                      stroke="#e7ebef"
                      strokeDasharray={band.key === "history" ? "5,4" : undefined}
                    />
                    <text
                      x={bounds.x + 10}
                      y={bounds.y + 16}
                      fontSize={8}
                      fontWeight={700}
                      letterSpacing={0.7}
                      fill={DAG_COLORS.muted}
                    >
                      {band.label.toUpperCase()}
                    </text>
                  </g>
                );
              })}

              {routedSegments.map((segment) => {
                const segmentMeta = topology.segmentMeta.get(segment.id);
                if (!segmentMeta) return null;
                const meta = topology.edgeMeta.get(segmentMeta.edgeId);
                if (!meta) return null;
                const visual = edgeVisual(meta);
                return (
                  <DagEdge
                    key={segment.id}
                    points={segment.points}
                    color={visual.color}
                    width={visual.width}
                    dashed={visual.disposition === "projected_edge"}
                    opacity={visual.opacity}
                    markerEnd={segmentMeta.markerEnd && !visual.activeClamp}
                    highlighted={hoveredEdge === meta.id}
                    onHoverChange={(hovered) => setHoveredEdge(hovered ? meta.id : null)}
                  />
                );
              })}

              {nodes.map((node) => {
                const meta = topology.nodeMeta.get(node.id);
                if (!meta) return null;
                if (meta.kind === "edge_slot") {
                  const edge = topology.edgeMeta.get(meta.edgeId);
                  if (!edge) return null;
                  const visual = edgeVisual(edge);
                  return (
                    <g
                      key={node.id}
                      transform={`translate(${node.x},${node.y})`}
                      onPointerEnter={() => setHoveredEdge(edge.id)}
                      onPointerLeave={() => setHoveredEdge(null)}
                    >
                      <EdgeSlot
                        meta={edge}
                        disposition={visual.disposition}
                        posterior={visual.posterior}
                        color={visual.color}
                        pruned={visual.activeClamp}
                        specificationVisible={specificationVisible}
                        dimmed={visual.dimmed}
                      />
                    </g>
                  );
                }

                const construct = meta.construct;
                const dimmed =
                  selectedNeighborhood != null && !selectedNeighborhood.has(construct.name);
                const selected = selectedNode === construct.name;
                const select = () =>
                  setSelectedNode((current) =>
                    current === construct.name ? null : construct.name,
                  );
                if (meta.kind === "history") {
                  return (
                    <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                      <HistoryCard
                        construct={construct}
                        status={nodeStatuses[construct.name]}
                        persistence={persistencePosteriors[construct.name]}
                        dimmed={dimmed}
                        selected={selected}
                        onSelect={select}
                      />
                    </g>
                  );
                }

                const nodeIndicators = indicatorsByConstruct.get(construct.name) ?? [];
                const reference = simulationResult
                  ? (getNodeReferenceSeries(simulationResult, construct.name) ?? [])
                  : [];
                const action = simulationResult
                  ? (getNodeActionSeries(simulationResult, construct.name) ?? [])
                  : [];
                const clamp =
                  currentDay == null
                    ? undefined
                    : simulationResult?.clamps.find(
                        (candidate) =>
                          candidate.variable === construct.name &&
                          candidate.from_day <= currentDay &&
                          (candidate.to_day == null || currentDay < candidate.to_day),
                      );
                const clampLabel = clamp ? `do(${clamp.mode})` : undefined;
                return (
                  <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                    <ConstructCard
                      construct={construct}
                      indicators={nodeIndicators}
                      likelihoodByVariable={likelihoodByVariable}
                      warningVariables={warningVariables}
                      status={nodeStatuses[construct.name]}
                      knownInput={knownInputNames.has(construct.name)}
                      persistence={persistencePosteriors[construct.name]}
                      days={days}
                      reference={reference}
                      action={action}
                      dayIndex={clampedDayIndex}
                      clampLabel={clampLabel}
                      selected={selected}
                      dimmed={dimmed}
                      onSelect={select}
                    />
                  </g>
                );
              })}
            </g>
          </DagSvg>
        )}
      </DagCanvasFrame>

      {simulationVisible && days.length > 0 ? (
        <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-white px-3 py-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-7 w-7 p-0"
            onClick={() => setPlaying((current) => !current)}
            aria-label={playing ? "Pause simulation timeline" : "Play simulation timeline"}
          >
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </Button>
          <input
            type="range"
            min={0}
            max={days.length - 1}
            value={clampedDayIndex}
            onChange={(event) => setDayIndex(Number(event.target.value))}
            className="min-w-52 flex-1 accent-blue-600"
            aria-label="Simulation day"
          />
          <span className="min-w-16 text-right font-mono text-[10px] text-slate-600">
            day {currentDay}
          </span>
          <span className="text-[10px] text-muted-foreground">
            gray reference · blue intervention
          </span>
        </div>
      ) : null}

      <div className="flex flex-wrap gap-x-4 gap-y-1 px-1 text-[9px] text-muted-foreground">
        <span>Lagged effects physically originate in t−1.</span>
        <span>Contemporaneous effects remain inside t.</span>
        {designVisible ? <span>Dashed edge slots are projected by design.</span> : null}
        {fitVisible ? <span>Edge color and weight show fitted sign and magnitude.</span> : null}
      </div>
    </div>
  );
}
