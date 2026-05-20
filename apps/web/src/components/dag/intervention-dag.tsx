"use client";

import { useElkLayout } from "@/lib/hooks/use-elk-layout";
import { useDagAnimation } from "@/lib/hooks/use-dag-animation";
import type { DagAnimationConfig } from "@/lib/hooks/use-dag-animation";
import type { CausalEdge, Construct, Indicator } from "@nof1-causal-lab/api-types";
import {
  Background,
  BackgroundVariant,
  type Edge,
  type EdgeTypes,
  type Node,
  type NodeChange,
  type NodeTypes,
  ReactFlow,
  applyNodeChanges,
} from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import type {
  EdgePosterior,
  Stage6SimulationResult,
} from "./intervention-dag-types";
import { AutoFitView } from "./auto-fit-view";
import { buildInterventionDagViewModel } from "./intervention-dag-view-model";
import { AnimationTimeline } from "./animation-timeline";
import { DAG_ZOOM_CONTROLS_INSET, DagZoomControls } from "./dag-zoom-controls";
import { EffectNode } from "./effect-node";
import { NoiseNode } from "./noise-node";
import { useMeasuredElement } from "./use-measured-element";
import { WeightedEdge } from "./weighted-edge";

// ── Props ─────────────────────────────────────────────────────────────

export interface InterventionDagProps {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  edgePosteriors?: Record<string, EdgePosterior>;
  processNoise?: Record<string, number>;
  showNoiseNodes?: boolean;
  requestedHorizonDays?: number;
  simulationResult?: Stage6SimulationResult | null;
  height?: string;
}

// ── Custom type registries ────────────────────────────────────────────

const nodeTypes: NodeTypes = {
  effect: EffectNode,
  noise: NoiseNode,
};

const edgeTypes: EdgeTypes = {
  weighted: WeightedEdge,
};

const NOISE_OFFSET_X = 20;
const NOISE_OFFSET_Y = -10;
const ASSUMED_NODE_WIDTH = 240;
const OVERLAY_GAP = 12;

// ── Component ─────────────────────────────────────────────────────────

export function InterventionDag({
  simulationResult,
  ...props
}: InterventionDagProps) {
  const animationKey = useMemo(
    () => (simulationResult ? JSON.stringify(simulationResult) : "static"),
    [simulationResult],
  );

  return (
    <InterventionDagCanvas
      key={animationKey}
      simulationResult={simulationResult}
      {...props}
    />
  );
}

function InterventionDagCanvas({
  constructs,
  edges,
  indicators,
  edgePosteriors = {},
  processNoise,
  showNoiseNodes = false,
  requestedHorizonDays,
  simulationResult = null,
  height = "600px",
}: InterventionDagProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // ── Layout ──────────────────────────────────────────────────────────
  const {
    nodes: layoutNodes,
    edges: flowEdges,
    isLayouting,
  } = useElkLayout(constructs, edges, indicators);

  // ── Animation config (memoised so the RAF effect doesn't re-fire) ──
  const animConfig: DagAnimationConfig | null = useMemo(() => {
    if (!simulationResult) {
      return null;
    }
    return {
      edges,
      constructs,
      result: simulationResult,
    };
  }, [edges, constructs, simulationResult]);

  const anim = useDagAnimation(animConfig);
  const viewModel = useMemo(
    () =>
      buildInterventionDagViewModel({
        constructs,
        requestedHorizonDays,
        result: simulationResult,
        animation: anim,
      }),
    [constructs, requestedHorizonDays, simulationResult, anim],
  );
  const mode = viewModel.mode;

  // ── Nodes ───────────────────────────────────────────────────────────
  const enrichedNodes: Node[] = useMemo(() => {
    const nodes: Node[] = layoutNodes.map((n) => ({
      ...n,
      type: "effect",
      data: {
        ...n.data,
        ...(viewModel.nodeData[n.id] ?? {}),
      },
    }));

    if (showNoiseNodes && processNoise) {
      for (const n of layoutNodes) {
        const variance = processNoise[n.id];
        if (variance == null) continue;
        nodes.push({
          id: `noise-${n.id}`,
          type: "noise",
          position: {
            x: (n.position?.x ?? 0) + ASSUMED_NODE_WIDTH + NOISE_OFFSET_X,
            y: (n.position?.y ?? 0) + NOISE_OFFSET_Y,
          },
          data: { constructName: n.id, variance },
          draggable: false,
          selectable: false,
          connectable: false,
        });
      }
    }

    return nodes;
  }, [layoutNodes, showNoiseNodes, processNoise, viewModel.nodeData]);

  // ── Edges ───────────────────────────────────────────────────────────
  const maxAbsPosterior = useMemo(() => {
    let max = 0;
    for (const p of Object.values(edgePosteriors)) {
      const a = Math.abs(p.mean);
      if (a > max) max = a;
    }
    return max;
  }, [edgePosteriors]);

  const enrichedEdges: Edge[] = useMemo(() => {
    return flowEdges.map((e) => {
      const key = `${e.source}\u2192${e.target}`;
      const posterior = edgePosteriors[key];
      const animState = mode === "static" ? "normal" : (anim.edgeStates[key] ?? "normal");

      return {
        ...e,
        type: "weighted" as const,
        data: { ...e.data, posterior, maxAbsPosterior, animState },
        // Arrowheads are rendered inside WeightedEdge with fixed size
        // (markerUnits="userSpaceOnUse") so they don't scale with strokeWidth.
        markerEnd: undefined,
      };
    });
  }, [flowEdges, edgePosteriors, maxAbsPosterior, anim.edgeStates, mode]);

  const fitViewKey = useMemo(
    () =>
      JSON.stringify(
        enrichedNodes.map((node) => [
          node.id,
          node.position?.x ?? 0,
          node.position?.y ?? 0,
        ]),
      ),
    [enrichedNodes],
  );
  const [timelineOverlayRef, timelineOverlaySize] = useMeasuredElement<HTMLDivElement>();
  const overlayInsets = useMemo(
    () => ({
      top: DAG_ZOOM_CONTROLS_INSET.top,
      right: 0,
      bottom: timelineOverlaySize.height > 0 ? timelineOverlaySize.height + OVERLAY_GAP : 0,
      left: DAG_ZOOM_CONTROLS_INSET.left,
    }),
    [timelineOverlaySize.height],
  );

  // ── Drag state (controlled mode) ───────────────────────────────────
  // localNodes holds React Flow's internal state (measured dimensions, drag
  // positions, etc.). We NEVER replace it with enrichedNodes except when the
  // node-id set changes (layout recalc / noise toggle). Animation data is
  // merged on top via the styledNodes derivation below.
  const [localNodes, setLocalNodes] = useState<Node[]>(enrichedNodes);
  const [prevKey, setPrevKey] = useState(() =>
    JSON.stringify(enrichedNodes.map((n) => n.id).sort()),
  );
  const key = JSON.stringify(enrichedNodes.map((n) => n.id).sort());

  if (key !== prevKey) {
    setPrevKey(key);
    setLocalNodes(enrichedNodes);
  }

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setLocalNodes((nds) => applyNodeChanges(changes, nds) as Node[]);
  }, []);

  // ── Merge live animation data onto stable local nodes ───────────────
  // Start from localNodes (preserves measured, position, React Flow internals)
  // and only patch in data + type from enrichedNodes.
  const styledNodes: Node[] = useMemo(() => {
    const dataMap = new Map(enrichedNodes.map((n) => [n.id, n]));

    let nodes: Node[] = localNodes
      .filter((n) => dataMap.has(n.id))
      .map((n) => {
        const enriched = dataMap.get(n.id)!;
        return { ...n, data: enriched.data, type: enriched.type };
      });

    // Append new nodes not yet in localNodes (freshly toggled noise nodes)
    const localIds = new Set(localNodes.map((n) => n.id));
    for (const n of enrichedNodes) {
      if (!localIds.has(n.id)) nodes.push(n);
    }

    // Selection highlighting
    if (selectedNode) {
      const connected = new Set<string>([selectedNode]);
      for (const e of enrichedEdges) {
        if (e.source === selectedNode || e.target === selectedNode) {
          connected.add(e.source);
          connected.add(e.target);
        }
      }
      nodes = nodes.map((n) => ({
        ...n,
        style: connected.has(n.id) ? {} : { opacity: 0.3 },
      }));
    }

    return nodes;
  }, [localNodes, enrichedNodes, enrichedEdges, selectedNode]);

  const styledEdges = useMemo(() => {
    if (!selectedNode) return enrichedEdges;
    return enrichedEdges.map((e) => ({
      ...e,
      style: {
        ...e.style,
        opacity:
          e.source === selectedNode || e.target === selectedNode ? 1 : 0.15,
      },
    }));
  }, [enrichedEdges, selectedNode]);

  // ── Handlers ────────────────────────────────────────────────────────
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: { id: string; type?: string }) => {
      if (node.type === "effect") {
        setSelectedNode((prev) => (prev === node.id ? null : node.id));
      }
    },
    [],
  );
  const handlePaneClick = useCallback(() => setSelectedNode(null), []);

  // ── Timeline markers ───────────────────────────────────────────────
  const timeStepsDays = viewModel.timeStepsDays;

  // ── Render ──────────────────────────────────────────────────────────
  if (isLayouting && layoutNodes.length === 0) {
    return <div className="w-full rounded-lg border bg-card" style={{ height }} />;
  }

  return (
    <div className="relative w-full overflow-hidden rounded-lg border bg-card" style={{ height }}>
      <div
        ref={timelineOverlayRef}
        className="absolute bottom-3 left-3 right-3 z-10"
      >
        {mode !== "static" && timeStepsDays.length > 0 ? (
          <AnimationTimeline
            isPlaying={anim.isPlaying}
            phase={anim.phase}
            timeStepsDays={timeStepsDays}
            currentTimeIndex={anim.timeIndex}
            temporalMarkers={viewModel.temporalMarkers}
            onPlay={anim.play}
            onPause={anim.pause}
            onReset={anim.reset}
            onScrub={anim.scrubTo}
          />
        ) : null}
      </div>
      <div className="h-full w-full">
        <ReactFlow
          nodes={styledNodes}
          edges={styledEdges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodesChange={onNodesChange}
          onNodeClick={handleNodeClick}
          onPaneClick={handlePaneClick}
          fitView
          fitViewOptions={{ padding: 0.25 }}
          nodesDraggable
          nodesConnectable={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
          <AutoFitView fitViewKey={fitViewKey} insets={overlayInsets} />
          <DagZoomControls />
        </ReactFlow>
      </div>
    </div>
  );
}
