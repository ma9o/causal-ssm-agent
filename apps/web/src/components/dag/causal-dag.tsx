"use client";

import { useElkLayout } from "@/lib/hooks/use-elk-layout";
import type { CausalEdge, Construct, Indicator } from "@causal-ssm/api-types";
import {
  Background,
  BackgroundVariant,
  type NodeChange,
  type NodeTypes,
  Panel,
  ReactFlow,
  applyNodeChanges,
} from "@xyflow/react";
import { useCallback, useMemo, useState } from "react";
import { ConstructNode } from "./construct-node";

interface CausalDagProps {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  onNodeClick?: (constructName: string) => void;
  height?: string;
}

const nodeTypes: NodeTypes = {
  construct: ConstructNode,
};

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

export function CausalDag({
  constructs,
  edges,
  indicators,
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

  const styledNodes = useMemo(() => {
    if (!selectedNode) return localNodes;
    const connectedIds = new Set<string>([selectedNode]);
    for (const e of flowEdges) {
      if (e.source === selectedNode || e.target === selectedNode) {
        connectedIds.add(e.source);
        connectedIds.add(e.target);
      }
    }
    return localNodes.map((n) => ({
      ...n,
      style: connectedIds.has(n.id) ? {} : { opacity: 0.3 },
    }));
  }, [localNodes, flowEdges, selectedNode]);

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
    <div className="w-full rounded-lg border bg-card" style={{ height }}>
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
        <Panel position="top-right">
          <EdgeLegend hasLagged={hasLagged} hasContemporaneous={hasContemporaneous} />
        </Panel>
      </ReactFlow>
    </div>
  );
}
