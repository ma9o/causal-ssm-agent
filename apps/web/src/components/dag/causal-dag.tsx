"use client";

import { useElkLayout } from "@/lib/hooks/use-elk-layout";
import type { CausalEdge, Construct, Indicator } from "@causal-ssm/api-types";
import {
  Background,
  BackgroundVariant,
  type NodeChange,
  type NodeTypes,
  ReactFlow,
  applyNodeChanges,
} from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import { AutoFitView } from "./auto-fit-view";
import type { ConstructStatus } from "./construct-node";
import { ConstructNode } from "./construct-node";
import { useMeasuredElement } from "./use-measured-element";

interface CausalDagProps {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  nodeStatuses?: Record<string, ConstructStatus>;
  onNodeClick?: (constructName: string) => void;
  height?: string;
}

const nodeTypes: NodeTypes = {
  construct: ConstructNode,
};

const OVERLAY_GAP = 12;

function EdgeLegend({
  hasLagged,
  hasContemporaneous,
}: { hasLagged: boolean; hasContemporaneous: boolean }) {
  return (
    <div className="rounded-md border bg-card/90 px-3 py-2 text-xs backdrop-blur-sm shadow-sm">
      <div className="flex items-center gap-4">
        {hasContemporaneous && (
          <div className="flex items-center gap-2">
            <svg width="28" height="8" className="shrink-0" aria-hidden="true">
              <line
                x1="0"
                y1="4"
                x2="28"
                y2="4"
                stroke="var(--edge-contemporary)"
                strokeWidth="2"
              />
              <polygon points="22,1 28,4 22,7" fill="var(--edge-contemporary)" />
            </svg>
            <span className="text-muted-foreground">same-time</span>
          </div>
        )}
        {hasLagged && (
          <div className="flex items-center gap-2">
            <svg width="28" height="8" className="shrink-0" aria-hidden="true">
              <line
                x1="0"
                y1="4"
                x2="28"
                y2="4"
                stroke="var(--edge-lagged)"
                strokeWidth="1.5"
                strokeDasharray="6,4"
              />
              <polygon points="22,1 28,4 22,7" fill="var(--edge-lagged)" />
            </svg>
            <span className="text-muted-foreground">lagged</span>
          </div>
        )}
      </div>
    </div>
  );
}

function NodeLegend({
  hasMarginalized,
  hasBlocking,
}: { hasMarginalized: boolean; hasBlocking: boolean }) {
  if (!hasMarginalized && !hasBlocking) return null;
  return (
    <div className="rounded-md border bg-card/90 px-3 py-2 text-xs backdrop-blur-sm shadow-sm">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded-sm border-2 border-foreground/50 bg-card" />
          <span className="text-muted-foreground">observed</span>
        </div>
        {hasMarginalized && (
          <div className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-sm border-2 border-success bg-card" />
            <span className="text-muted-foreground">marginalized</span>
          </div>
        )}
        {hasBlocking && (
          <div className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-sm border-2 border-destructive bg-card" />
            <span className="text-muted-foreground">blocking</span>
          </div>
        )}
      </div>
    </div>
  );
}

export function CausalDag({
  constructs,
  edges,
  indicators,
  nodeStatuses,
  onNodeClick,
  height = "500px",
}: CausalDagProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const {
    nodes: layoutNodes,
    edges: flowEdges,
    isLayouting,
  } = useElkLayout(constructs, edges, indicators);

  // Local node state so dragging works (React Flow controlled mode needs onNodesChange)
  const [localNodes, setLocalNodes] = useState(layoutNodes);
  const [prevNodeKey, setPrevNodeKey] = useState(() =>
    JSON.stringify(layoutNodes.map((n) => n.id)),
  );
  const nodeKey = JSON.stringify(layoutNodes.map((n) => n.id));

  // Sync external layout changes into local drag state (React derive-state-from-props pattern)
  if (nodeKey !== prevNodeKey) {
    setPrevNodeKey(nodeKey);
    setLocalNodes(layoutNodes);
  }

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setLocalNodes((nds) => applyNodeChanges(changes, nds));
  }, []);

  const hasLagged = edges.some((e) => e.lagged);
  const hasContemporaneous = edges.some((e) => !e.lagged);
  const hasMarginalized = !!nodeStatuses && Object.values(nodeStatuses).includes("marginalized");
  const hasBlocking = !!nodeStatuses && Object.values(nodeStatuses).includes("blocking");
  const showEdgeLegend = onNodeClick != null || nodeStatuses != null;
  const showNodeLegend = hasMarginalized || hasBlocking;
  const showTopLegend = showEdgeLegend || showNodeLegend;

  // Merge node statuses into node data
  const nodesWithStatus = useMemo(() => {
    if (!nodeStatuses) return localNodes;
    return localNodes.map((n) => ({
      ...n,
      data: { ...n.data, status: nodeStatuses[n.id] },
    }));
  }, [localNodes, nodeStatuses]);

  const fitViewKey = useMemo(
    () =>
      JSON.stringify(
        layoutNodes.map((node) => [
          node.id,
          node.position?.x ?? 0,
          node.position?.y ?? 0,
        ]),
      ),
    [layoutNodes],
  );
  const [legendOverlayRef, legendOverlaySize] = useMeasuredElement<HTMLDivElement>();
  const overlayInsets = useMemo(
    () => ({
      top: showTopLegend && legendOverlaySize.height > 0 ? legendOverlaySize.height + OVERLAY_GAP : 0,
      right: 0,
      bottom: 0,
      left: 0,
    }),
    [legendOverlaySize.height, showTopLegend],
  );

  const styledNodes = useMemo(() => {
    if (!selectedNode) return nodesWithStatus;
    const connectedIds = new Set<string>([selectedNode]);
    for (const e of flowEdges) {
      if (e.source === selectedNode || e.target === selectedNode) {
        connectedIds.add(e.source);
        connectedIds.add(e.target);
      }
    }
    return nodesWithStatus.map((n) => ({
      ...n,
      style: connectedIds.has(n.id) ? {} : { opacity: 0.3 },
    }));
  }, [nodesWithStatus, flowEdges, selectedNode]);

  const styledEdges = useMemo(() => {
    if (!selectedNode) return flowEdges;
    return flowEdges.map((e) => ({
      ...e,
      style: {
        ...e.style,
        opacity: e.source === selectedNode || e.target === selectedNode ? 1 : 0.15,
      },
    }));
  }, [flowEdges, selectedNode]);

  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: { id: string; type?: string }) => {
      if (node.type === "construct") {
        setSelectedNode((prev) => (prev === node.id ? null : node.id));
        onNodeClick?.(node.id);
      }
    },
    [onNodeClick],
  );

  const handlePaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  if (isLayouting && localNodes.length === 0) {
    return <div className="w-full rounded-lg border bg-card" style={{ height }} />;
  }

  return (
    <div className="relative w-full overflow-hidden rounded-lg border bg-card" style={{ height }}>
      <ReactFlow
        nodes={styledNodes}
        edges={styledEdges}
        nodeTypes={nodeTypes}
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
        defaultEdgeOptions={{
          style: { strokeWidth: 2 },
        }}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        <AutoFitView fitViewKey={fitViewKey} insets={overlayInsets} />
      </ReactFlow>
      {showTopLegend ? (
        <div
          ref={legendOverlayRef}
          className="pointer-events-none absolute right-3 top-3 z-10 flex flex-col items-end gap-2"
        >
          {showEdgeLegend ? (
            <EdgeLegend hasLagged={hasLagged} hasContemporaneous={hasContemporaneous} />
          ) : null}
          <NodeLegend
            hasMarginalized={hasMarginalized}
            hasBlocking={hasBlocking}
          />
        </div>
      ) : null}
    </div>
  );
}
