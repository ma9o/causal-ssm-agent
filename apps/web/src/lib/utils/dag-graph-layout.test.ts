import type { ElkNode } from "elkjs/lib/elk.bundled.js";
import { describe, expect, it } from "vitest";
import { buildElkGraph, readElkLayout } from "./dag-graph-layout";

describe("buildElkGraph", () => {
  it("maps nodes and edges and defaults the direction to RIGHT", () => {
    const graph = buildElkGraph({
      nodes: [
        { id: "a", width: 100, height: 50 },
        { id: "b", width: 100, height: 50 },
      ],
      edges: [{ id: "e0", source: "a", target: "b" }],
    });

    expect(graph.id).toBe("root");
    expect(graph.layoutOptions?.["elk.direction"]).toBe("RIGHT");
    expect(graph.layoutOptions?.["elk.edgeRouting"]).toBe("ORTHOGONAL");
    expect(graph.children).toHaveLength(2);
    expect(graph.children?.[0]).toMatchObject({ id: "a", width: 100, height: 50 });
    expect(graph.edges).toEqual([{ id: "e0", sources: ["a"], targets: ["b"] }]);
  });

  it("honours direction and merges graph + per-node layout options", () => {
    const graph = buildElkGraph({
      nodes: [
        { id: "a", width: 1, height: 1, layoutOptions: { "elk.partitioning.partition": "2" } },
      ],
      edges: [],
      direction: "DOWN",
      layoutOptions: { "elk.partitioning.activate": "true" },
    });

    expect(graph.layoutOptions?.["elk.direction"]).toBe("DOWN");
    expect(graph.layoutOptions?.["elk.partitioning.activate"]).toBe("true");
    expect(graph.children?.[0].layoutOptions?.["elk.partitioning.partition"]).toBe("2");
  });
});

describe("readElkLayout", () => {
  it("reads node geometry and routed edge polylines", () => {
    const laidOut: ElkNode = {
      id: "root",
      width: 300,
      height: 120,
      children: [
        { id: "a", x: 0, y: 0, width: 100, height: 50 },
        { id: "b", x: 200, y: 60, width: 100, height: 50 },
      ],
      edges: [
        {
          id: "e0",
          sources: ["a"],
          targets: ["b"],
          sections: [
            {
              id: "s0",
              startPoint: { x: 100, y: 25 },
              bendPoints: [{ x: 150, y: 25 }],
              endPoint: { x: 200, y: 85 },
            },
          ],
        },
      ],
    };

    const result = readElkLayout(laidOut);
    expect(result.width).toBe(300);
    expect(result.height).toBe(120);
    expect(result.nodes).toEqual([
      { id: "a", x: 0, y: 0, width: 100, height: 50 },
      { id: "b", x: 200, y: 60, width: 100, height: 50 },
    ]);
    expect(result.edges).toHaveLength(1);
    expect(result.edges[0]).toMatchObject({ id: "e0", source: "a", target: "b" });
    expect(result.edges[0].points).toEqual([
      { x: 100, y: 25 },
      { x: 150, y: 25 },
      { x: 200, y: 85 },
    ]);
  });

  it("yields an empty polyline when an edge has no routed section", () => {
    const result = readElkLayout({
      id: "root",
      children: [],
      edges: [{ id: "e0", sources: ["a"], targets: ["b"] }],
    });
    expect(result.edges[0].points).toEqual([]);
  });
});
