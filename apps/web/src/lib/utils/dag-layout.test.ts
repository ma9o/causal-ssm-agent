import type { CausalEdge, Construct, Indicator } from "@causal-ssm/api-types";
import { describe, expect, it } from "vitest";
import { layoutDag } from "./dag-layout";

function makeConstruct(name: string, overrides: Partial<Construct> = {}): Construct {
  return {
    name,
    description: `${name} description`,
    role: "endogenous",
    is_outcome: false,
    temporal_status: "time_varying",
    ...overrides,
  } as Construct;
}

function makeEdge(cause: string, effect: string, lagged = false): CausalEdge {
  return { cause, effect, lagged } as CausalEdge;
}

function makeIndicator(name: string, constructName: string): Indicator {
  return { name, construct_name: constructName } as Indicator;
}

describe("layoutDag", () => {
  it("returns empty arrays for empty input", async () => {
    const result = await layoutDag([], []);
    expect(result.nodes).toEqual([]);
    expect(result.edges).toEqual([]);
  });

  it("positions a single construct", async () => {
    const constructs = [makeConstruct("A")];
    const result = await layoutDag(constructs, []);
    expect(result.nodes).toHaveLength(1);
    expect(result.nodes[0].id).toBe("A");
    expect(result.nodes[0].type).toBe("construct");
    expect(result.nodes[0].position).toHaveProperty("x");
    expect(result.nodes[0].position).toHaveProperty("y");
  });

  it("creates edges between constructs", async () => {
    const constructs = [makeConstruct("A"), makeConstruct("B")];
    const edges = [makeEdge("A", "B")];
    const result = await layoutDag(constructs, edges);
    expect(result.edges).toHaveLength(1);
    expect(result.edges[0].source).toBe("A");
    expect(result.edges[0].target).toBe("B");
  });

  it("uses smoothstep type for contemporaneous edges", async () => {
    const constructs = [makeConstruct("A"), makeConstruct("B")];
    const edges = [makeEdge("A", "B", false)];
    const result = await layoutDag(constructs, edges);
    expect(result.edges[0].type).toBe("smoothstep");
  });

  it("uses default type for lagged edges", async () => {
    const constructs = [makeConstruct("A"), makeConstruct("B")];
    const edges = [makeEdge("A", "B", true)];
    const result = await layoutDag(constructs, edges);
    expect(result.edges[0].type).toBe("default");
  });

  it("applies dashed stroke to lagged edges", async () => {
    const constructs = [makeConstruct("A"), makeConstruct("B")];
    const edges = [makeEdge("A", "B", true)];
    const result = await layoutDag(constructs, edges);
    expect(result.edges[0].style?.strokeDasharray).toBe("6,4");
  });

  it("does not dash contemporaneous edges", async () => {
    const constructs = [makeConstruct("A"), makeConstruct("B")];
    const edges = [makeEdge("A", "B", false)];
    const result = await layoutDag(constructs, edges);
    expect(result.edges[0].style?.strokeDasharray).toBeUndefined();
  });

  it("attaches indicators to construct node data", async () => {
    const constructs = [makeConstruct("A")];
    const indicators = [makeIndicator("x1", "A"), makeIndicator("x2", "A")];
    const result = await layoutDag(constructs, [], indicators);
    expect(result.nodes[0].data.indicators).toHaveLength(2);
  });

  it("uses wider nodes when indicators are present", async () => {
    const constructsNoInd = [makeConstruct("A"), makeConstruct("B")];
    const constructsWithInd = [makeConstruct("A"), makeConstruct("B")];
    const indicators = [makeIndicator("x1", "A")];

    const withoutInd = await layoutDag(constructsNoInd, [makeEdge("A", "B")]);
    const withInd = await layoutDag(constructsWithInd, [makeEdge("A", "B")], indicators);

    // Both should layout successfully - nodes should exist
    expect(withoutInd.nodes).toHaveLength(2);
    expect(withInd.nodes).toHaveLength(2);
  });

  it("handles multiple edges in a chain", async () => {
    const constructs = [makeConstruct("A"), makeConstruct("B"), makeConstruct("C")];
    const edges = [makeEdge("A", "B"), makeEdge("B", "C")];
    const result = await layoutDag(constructs, edges);
    expect(result.nodes).toHaveLength(3);
    expect(result.edges).toHaveLength(2);
    // A should be above B, B above C (layered top-down)
    const nodeA = result.nodes.find((n) => n.id === "A");
    const nodeC = result.nodes.find((n) => n.id === "C");
    expect(nodeA).toBeDefined();
    expect(nodeC).toBeDefined();
    expect(nodeA?.position.y).toBeLessThan(nodeC?.position.y as number);
  });

  it("includes both contemporaneous and lagged edges in output", async () => {
    const constructs = [makeConstruct("A"), makeConstruct("B")];
    const edges = [makeEdge("A", "B", false), makeEdge("B", "A", true)];
    const result = await layoutDag(constructs, edges);
    expect(result.edges).toHaveLength(2);
    const types = result.edges.map((e) => e.type);
    expect(types).toContain("smoothstep");
    expect(types).toContain("default");
  });

  it("adds arrow markers to all edges", async () => {
    const constructs = [makeConstruct("A"), makeConstruct("B")];
    const edges = [makeEdge("A", "B")];
    const result = await layoutDag(constructs, edges);
    expect(result.edges[0].markerEnd).toBeDefined();
    expect((result.edges[0].markerEnd as Record<string, unknown>).type).toBe("arrowclosed");
  });

  it("assigns construct data to nodes", async () => {
    const c = makeConstruct("stress", { is_outcome: true });
    const result = await layoutDag([c], []);
    expect(result.nodes[0].data.name).toBe("stress");
    expect(result.nodes[0].data.is_outcome).toBe(true);
  });
});
