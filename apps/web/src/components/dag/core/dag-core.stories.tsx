import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useMemo, useState } from "react";
import { DagEdge } from "./dag-edge";
import { DagNodeShell } from "./dag-node";
import { DAG_COLORS } from "./palette";

interface DemoNode {
  id: string;
  title: string;
  subtitle: string;
  outcome?: boolean;
}

const NODES: DemoNode[] = [
  { id: "life_events_load", title: "life events load", subtitle: "exo · held" },
  { id: "adherence", title: "adherence", subtitle: "endo · varying" },
  { id: "serotonergic_exposure", title: "serotonergic exposure", subtitle: "endo · varying" },
  { id: "physical_activity", title: "physical activity", subtitle: "endo · varying" },
  { id: "sleep_quality", title: "sleep quality", subtitle: "endo · varying" },
  { id: "affective_state", title: "affective state", subtitle: "endo · varying", outcome: true },
];

const EDGES: Array<[string, string]> = [
  ["adherence", "serotonergic_exposure"],
  ["life_events_load", "affective_state"],
  ["serotonergic_exposure", "sleep_quality"],
  ["serotonergic_exposure", "affective_state"],
  ["physical_activity", "sleep_quality"],
  ["sleep_quality", "affective_state"],
];

const NODE_W = 208;
const NODE_H = 60;
const META: Record<string, DemoNode> = Object.fromEntries(NODES.map((n) => [n.id, n]));

/**
 * Smoke-test harness for the shared bespoke DAG core: build a small causal
 * graph, lay it out with ELK (routed), and render it through the edge + node
 * primitives. Hover an edge to light it and its endpoints.
 */
function DagCoreDemo() {
  const graph: DagGraphInput = useMemo(
    () => ({
      nodes: NODES.map((n) => ({ id: n.id, width: NODE_W, height: NODE_H })),
      edges: EDGES.map(([source, target], i) => ({ id: `e${i}`, source, target })),
      direction: "RIGHT",
    }),
    [],
  );

  const { nodes, edges, width, height, isLayouting } = useDagLayout(graph);
  const [hovered, setHovered] = useState<string | null>(null);

  const hoveredEdge = edges.find((e) => e.id === hovered) ?? null;
  const litNodes = new Set(hoveredEdge ? [hoveredEdge.source, hoveredEdge.target] : []);

  if (isLayouting) {
    return <div className="h-[560px] w-full rounded-lg border bg-card" />;
  }

  return (
    <div className="h-[560px] w-full overflow-auto rounded-lg border bg-card p-6">
      <svg
        width={Math.ceil(width)}
        height={Math.ceil(height)}
        viewBox={`0 0 ${Math.ceil(width)} ${Math.ceil(height)}`}
        role="img"
        aria-label="DAG core renderer demo"
        className="block"
      >
        {edges.map((e) => {
          const lit = e.id === hovered;
          return (
            <DagEdge
              key={e.id}
              points={e.points}
              color={lit ? DAG_COLORS.positive : DAG_COLORS.contemporaneous}
              highlighted={lit}
              onHoverChange={(h) => setHovered(h ? e.id : null)}
            />
          );
        })}
        {nodes.map((n) => {
          const node = META[n.id];
          const lit = litNodes.has(n.id);
          return (
            <g key={n.id} transform={`translate(${n.x} ${n.y})`}>
              <DagNodeShell
                width={n.width}
                height={n.height}
                title={node?.title}
                subtitle={node?.subtitle}
                accent={lit ? DAG_COLORS.positive : undefined}
                highlighted={lit}
                outcome={node?.outcome}
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

const meta: Meta<typeof DagCoreDemo> = {
  title: "DAG/Core/Bespoke renderer",
  component: DagCoreDemo,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof DagCoreDemo>;

export const StructureGraph: Story = {};
