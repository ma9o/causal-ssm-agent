"use client";

import type { AnalysisStageRun } from "@/lib/api/analysis";
import type { Stage4BlockNodeData, Stage4BlockStatus } from "@/components/dag/stage4-block-node";
import { Stage4BlockNode } from "@/components/dag/stage4-block-node";
import { Stage4SectionEdge } from "@/components/dag/stage4-section-edge";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import {
  STAGE4_DONE_NODE_ID,
  STAGE4_LOCK_NODE_ID,
  STAGE4_REPAIR_BARRIER_NODE_ID,
  type Stage4BlockLastState,
  type Stage4Graph,
  type Stage4Snapshot,
  useStage4Graph,
} from "@/lib/hooks/use-stage4-graph";
import {
  Background,
  BackgroundVariant,
  type Edge,
  type EdgeTypes,
  type Node,
  type NodeTypes,
  ReactFlow,
} from "@xyflow/react";
import { CheckCircle2, Loader2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  deriveStage4SectionEdges,
  getStage4SectionId,
  isOptionalStage4Section,
  routeStage4SectionEdge,
  STAGE4_SECTION_NODE_POSITION,
  STAGE4_SECTION_ORDER,
  type Stage4SectionId,
} from "@/lib/stage4-section-graph";

interface Stage4SectionSummary {
  id: Stage4SectionId;
  label: string;
  tooltip: string;
  displayLabel: string;
  detailLabel: string;
  status: Stage4BlockStatus;
  isActive: boolean;
  totalCount: number;
  acceptedCount: number;
  reopenedCount: number;
  statusItems: NonNullable<Stage4BlockNodeData["statusItems"]>;
}

interface Stage4LayoutResult {
  nodes: Node[];
  edges: Edge[];
  width: number;
  height: number;
}

const nodeTypes: NodeTypes = {
  stage4Block: Stage4BlockNode,
};
const edgeTypes: EdgeTypes = {
  stage4Section: Stage4SectionEdge,
};

function formatStage4Value(value: unknown): string {
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    if (value === 0) return "0";
    const magnitude = Math.abs(value);
    if (magnitude >= 1000 || magnitude < 0.001) return value.toPrecision(2);
    return Number.isInteger(value)
      ? String(value)
      : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => formatStage4Value(item)).join(", ")}]`;
  }
  if (value === null) return "null";
  if (value === undefined) return "";
  return String(value);
}

function formatStage4DistributionCall(
  distribution?: string,
  params?: Record<string, unknown>,
): string {
  if (!distribution) return "";
  if (!params || Object.keys(params).length === 0) return distribution;
  const rendered = Object.entries(params)
    .map(([key, value]) => `${key}=${formatStage4Value(value)}`)
    .join(", ");
  return `${distribution}(${rendered})`;
}

function formatStage4LastBlockStateDetail(lastState: Stage4BlockLastState | undefined): string {
  if (!lastState) return "";
  switch (lastState.detail_kind) {
    case "revision":
      return lastState.reason ?? "";
    case "review_approval":
      return lastState.reasoning ?? "";
    case "indicator_choice": {
      const head =
        lastState.distribution && lastState.link
          ? `${lastState.distribution} with ${lastState.link} link`
          : (lastState.distribution ?? "");
      if (!head) return lastState.reasoning ?? "";
      return lastState.reasoning ? `${head}. ${lastState.reasoning}` : head;
    }
    case "prior_bundle": {
      const priors = lastState.priors ?? [];
      if (priors.length === 0) return "";
      const single = priors.length === 1;
      return priors
        .map((prior) => {
          const call = formatStage4DistributionCall(prior.distribution, prior.params);
          return single ? call : `${prior.parameter} ~ ${call}`;
        })
        .filter((value) => value.length > 0)
        .join("; ");
    }
    default:
      return "";
  }
}

function buildSectionSummaries(
  graph: Stage4Graph | null,
  snapshot: Stage4Snapshot | null,
  lastBlockStateById: Record<string, Stage4BlockLastState>,
): Stage4SectionSummary[] {
  if (!graph) return [];

  const currentNodeId =
    snapshot?.cursor.kind === "block"
      ? snapshot.cursor.block_id
      : snapshot?.cursor.kind === "model_spec_lock"
        ? STAGE4_LOCK_NODE_ID
        : snapshot?.cursor.kind === "repair_barrier"
          ? STAGE4_REPAIR_BARRIER_NODE_ID
          : snapshot?.cursor.kind === "done"
            ? STAGE4_DONE_NODE_ID
            : null;

  const repairScopeIds = new Set(
    snapshot?.repair_campaign?.scope_block_ids ??
      (snapshot?.cursor.kind === "repair_barrier" ? snapshot.cursor.scope_block_ids : []),
  );
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));

  return STAGE4_SECTION_ORDER.map((section) => {
    const sectionNodes = graph.nodes.filter((node) => getStage4SectionId(node.kind) === section.id);
    const logicalNodes = sectionNodes.filter((node) => !node.id.startsWith("__"));
    const activeNode = currentNodeId ? byId.get(currentNodeId) : null;
    const activeInSection = Boolean(
      activeNode && getStage4SectionId(activeNode.kind) === section.id,
    );

    const statusItems = logicalNodes.map((node) => ({
      id: node.id,
      label: node.label,
      status: (snapshot?.block_status[node.id] ?? "pending") as Stage4BlockStatus,
      isActive: node.id === currentNodeId,
      inRepairScope: repairScopeIds.has(node.id),
      detailText: formatStage4LastBlockStateDetail(lastBlockStateById[node.id]),
    }));

    const totalCount = logicalNodes.length;
    const acceptedCount = statusItems.filter((item) => item.status === "accepted").length;
    const reopenedCount = statusItems.filter((item) => item.status === "reopened").length;
    const allInactive = totalCount > 0 && statusItems.every((item) => item.status === "inactive");
    const optionalAbsent = totalCount === 0 && isOptionalStage4Section(section.id);
    const nextOpenItem =
      statusItems.find((item) => item.status === "reopened") ??
      statusItems.find((item) => item.status === "pending") ??
      statusItems[0];
    const acceptedTailItem = [...statusItems].reverse().find((item) => item.status === "accepted");
    const activeStatusItem = statusItems.find((item) => item.id === activeNode?.id);

    let status: Stage4BlockStatus = "pending";
    if (activeInSection) {
      status =
        activeNode?.id && !activeNode.id.startsWith("__")
          ? ((snapshot?.block_status[activeNode.id] ?? "pending") as Stage4BlockStatus)
          : "pending";
    } else if (section.id === "done") {
      status = snapshot?.cursor.kind === "done" ? "accepted" : "pending";
    } else if (section.id === "repair_barrier") {
      status = snapshot?.cursor.kind === "repair_barrier" ? "accepted" : "pending";
    } else if (optionalAbsent) {
      status = "inactive";
    } else if (allInactive) {
      status = "inactive";
    } else if (reopenedCount > 0) {
      status = "reopened";
    } else if (totalCount > 0 && acceptedCount === totalCount) {
      status = "accepted";
    }

    let displayLabel = section.label;
    let detailLabel = "";

    if (section.id === "done") {
      displayLabel = snapshot?.cursor.kind === "done" ? "Stage 4 complete" : "Awaiting completion";
    } else if (section.id === "repair_barrier") {
      displayLabel =
        snapshot?.cursor.kind === "repair_barrier" ? "Validate repair scope" : "Repair barrier";
      detailLabel =
        snapshot?.cursor.kind === "repair_barrier"
          ? `${repairScopeIds.size} repaired block${repairScopeIds.size === 1 ? "" : "s"} ready`
          : "";
    } else if (activeInSection && activeNode) {
      displayLabel = activeNode.label;
      detailLabel = activeStatusItem?.detailText ?? "";
    } else if (status === "accepted" && totalCount > 0) {
      displayLabel = logicalNodes[logicalNodes.length - 1]?.label ?? section.label;
      detailLabel = acceptedTailItem?.detailText ?? "";
    } else if (reopenedCount > 0 && nextOpenItem) {
      displayLabel = nextOpenItem.label;
      detailLabel = nextOpenItem.detailText ?? `${reopenedCount} reopened`;
    } else if (nextOpenItem) {
      displayLabel = nextOpenItem.label;
    } else if (optionalAbsent) {
      detailLabel = "Not required for this plan";
    }

    return {
      id: section.id,
      label: section.label,
      tooltip: section.tooltip,
      displayLabel,
      detailLabel,
      status,
      isActive: activeInSection,
      totalCount,
      acceptedCount,
      reopenedCount,
      statusItems,
    };
  });
}

function layoutSectionGraph(
  sections: Stage4SectionSummary[],
  sectionEdges: ReturnType<typeof deriveStage4SectionEdges>,
  snapshot: Stage4Snapshot | null,
): Stage4LayoutResult {
  const nodes: Node[] = sections.map((section) => ({
    id: section.id,
    type: "stage4Block",
    position: STAGE4_SECTION_NODE_POSITION[section.id],
    data: {
      id: section.id,
      kind: section.id,
      label: section.displayLabel,
      phase: section.id,
      status: section.status,
      isActive: section.isActive,
      sectionLabel: section.label,
      totalCount: section.totalCount,
      acceptedCount: section.acceptedCount,
      reopenedCount: section.reopenedCount,
      statusItems: section.statusItems,
      detailLabel: section.detailLabel,
      tooltipText: section.tooltip,
    } satisfies Stage4BlockNodeData,
  }));

  const acceptedSections = new Set(
    sections.filter((section) => section.status === "accepted").map((section) => section.id),
  );
  const activeSectionId = sections.find((section) => section.isActive)?.id ?? null;
  const repairLive =
    Boolean(snapshot?.repair_campaign) ||
    snapshot?.cursor.kind === "repair_barrier" ||
    activeSectionId === "global_prior_review";

  const edges: Edge[] = sectionEdges.map((edge, index) => {
    const isRepair = edge.kind === "repair_transition";
    const sourceLive = acceptedSections.has(edge.from) || edge.from === activeSectionId;
    const targetLive = acceptedSections.has(edge.to) || edge.to === activeSectionId;
    const isActiveTransition =
      activeSectionId !== null && edge.to === activeSectionId && acceptedSections.has(edge.from);
    const isTraversed = sourceLive && targetLive;

    let opacity = 0.18;
    if (isRepair) {
      opacity = repairLive ? 0.68 : 0.12;
    } else if (isActiveTransition) {
      opacity = 0.95;
    } else if (isTraversed) {
      opacity = 0.72;
    } else {
      opacity = 0.3;
    }

    return {
      id: `stage4-section-edge-${index}`,
      source: edge.from,
      target: edge.to,
      type: "stage4Section",
      animated: isActiveTransition,
      data: {
        kind: edge.kind,
        points: routeStage4SectionEdge(edge.from, edge.to),
      },
      style: {
        stroke: isRepair ? "var(--edge-lagged)" : "var(--edge-contemporary)",
        strokeWidth: isActiveTransition ? 2.6 : edge.kind === "phase_advance" ? 2.1 : 1.7,
        strokeDasharray: isRepair ? "6,4" : undefined,
        opacity,
      },
    };
  });

  return {
    nodes,
    edges,
    width: 1088,
    height: 760,
  };
}

export function Stage4RunningView({
  graph,
  snapshot,
  lastBlockStateById = {},
}: {
  graph: Stage4Graph | null;
  snapshot: Stage4Snapshot | null;
  lastBlockStateById?: Record<string, Stage4BlockLastState>;
}) {
  const sections = useMemo(
    () => buildSectionSummaries(graph, snapshot, lastBlockStateById),
    [graph, lastBlockStateById, snapshot],
  );
  const sectionEdges = useMemo(() => deriveStage4SectionEdges(graph), [graph]);
  const baseLayout = useMemo(
    () => layoutSectionGraph(sections, sectionEdges, snapshot),
    [sections, sectionEdges, snapshot],
  );
  const [rowMinHeights, setRowMinHeights] = useState<Record<string, number>>({});
  const containerRef = useRef<HTMLDivElement>(null);
  const readyRef = useRef(false);

  const measureRows = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const nodeEls = container.querySelectorAll<HTMLElement>('[data-testid^="rf__node-"]');
    if (nodeEls.length === 0) return;

    const viewport = container.querySelector<HTMLElement>(".react-flow__viewport");
    const scaleMatch = viewport?.style.transform?.match(/scale\(([^)]+)\)/);
    const scale = scaleMatch ? parseFloat(scaleMatch[1]) : 1;

    const rowMap = new Map<number, { id: string; height: number }[]>();
    for (const el of nodeEls) {
      const id = el.dataset.id;
      if (!id) continue;
      const pos = STAGE4_SECTION_NODE_POSITION[id as Stage4SectionId];
      if (!pos) continue;
      const height = el.getBoundingClientRect().height / scale;
      const row = rowMap.get(pos.y) ?? [];
      row.push({ id, height });
      rowMap.set(pos.y, row);
    }

    const newMinHeights: Record<string, number> = {};
    for (const [, nodes] of rowMap) {
      const maxH = Math.ceil(Math.max(...nodes.map((n) => n.height)));
      for (const node of nodes) {
        newMinHeights[node.id] = maxH;
      }
    }
    setRowMinHeights((prev) => {
      if (JSON.stringify(prev) === JSON.stringify(newMinHeights)) return prev;
      return newMinHeights;
    });
  }, []);

  const onInit = useCallback(() => {
    readyRef.current = true;
    requestAnimationFrame(measureRows);
  }, [measureRows]);

  // Re-measure when sections change (e.g. animation steps)
  useEffect(() => {
    if (!readyRef.current) return;
    requestAnimationFrame(measureRows);
  }, [sections, measureRows]);

  const layout = useMemo(() => {
    if (Object.keys(rowMinHeights).length === 0) return baseLayout;
    return {
      ...baseLayout,
      nodes: baseLayout.nodes.map((node) => {
        const minHeight = rowMinHeights[node.id];
        if (!minHeight) return node;
        return { ...node, data: { ...node.data, minHeight } };
      }),
    };
  }, [baseLayout, rowMinHeights]);

  const visibleBlockStatuses = snapshot
    ? Object.values(snapshot.block_status).filter((status) => status !== "inactive")
    : [];
  const totalBlocks =
    visibleBlockStatuses.length > 0
      ? visibleBlockStatuses.length
      : graph
        ? graph.nodes.filter(
            (node) => !node.id.startsWith("__") && node.kind !== "global_prior_review",
          ).length
        : 0;
  const acceptedBlocks = visibleBlockStatuses.filter((status) => status === "accepted").length;

  if (!graph || graph.nodes.length === 0) {
    return (
      <div className="flex items-center gap-2 py-3 text-sm text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Building model specification plan...
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5 font-medium tabular-nums text-foreground">
          {snapshot?.cursor.kind === "done" ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
          ) : (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
          )}
          {acceptedBlocks}/{totalBlocks} blocks
        </span>
      </div>

      <div ref={containerRef} className="relative w-full overflow-hidden rounded-xl border bg-card">
        <div className="relative h-[760px] w-full">
          <ReactFlow
            nodes={layout.nodes}
            edges={layout.edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            defaultViewport={{ x: 10, y: 0, zoom: 1 }}
            fitView
            fitViewOptions={{ padding: 0.12, minZoom: 0.75, maxZoom: 1 }}
            onInit={onInit}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            zoomOnScroll={false}
            zoomOnPinch={false}
            zoomOnDoubleClick={false}
            panOnDrag={false}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={18} size={1} />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}

export default function Stage4RunningContent({
  workspaceId,
  stageStatus,
  stageRun,
}: {
  workspaceId: string;
  stageStatus: StageRunStatus;
  stageRun?: AnalysisStageRun | null;
}) {
  const { graph, snapshot, lastBlockStateById } = useStage4Graph(
    workspaceId,
    stageStatus,
    stageRun?.ownerRootFlowRunId ?? null,
  );
  return (
    <Stage4RunningView graph={graph} snapshot={snapshot} lastBlockStateById={lastBlockStateById} />
  );
}
