import { describe, expect, it } from "vitest";
import { constructs, edges } from "@/components/dag/__fixtures__/dag-base-fixtures";
import { buildSimulationGraph } from "./build-cone-graph";

describe("buildSimulationGraph", () => {
  it("separates DEMO cross-lagged edges from contemporaneous and fitted persistence edges", () => {
    const built = buildSimulationGraph(constructs, edges, {
      dir: "RIGHT",
      showIndicators: false,
      showUnroll: true,
      indicators: [],
      persistenceNodes: ["internalizing_symptom_burden"],
    });
    const metadata = [...built.edgeMeta.values()];

    expect(metadata).toContainEqual({
      a: "internalizing_symptom_burden__p",
      b: "patient_taper_preference_beliefs",
      isSelf: false,
      lagged: true,
    });
    expect(metadata).toContainEqual({
      a: "external_stressful_events",
      b: "perceived_stress_burden",
      isSelf: false,
      lagged: false,
    });
    expect(metadata).toContainEqual({
      a: "natural_recovery_propensity",
      b: "internalizing_symptom_burden",
      isSelf: false,
      lagged: true,
    });
    expect(metadata).toContainEqual({
      a: "internalizing_symptom_burden__p",
      b: "internalizing_symptom_burden",
      isSelf: true,
      lagged: true,
    });
    expect(built.graph.nodes.some(({ id }) => id === "internalizing_symptom_burden__p")).toBe(true);
    expect(built.graph.nodes.some(({ id }) => id === "external_stressful_events__p")).toBe(false);
    expect(built.graph.nodes.some(({ id }) => id === "natural_recovery_propensity__p")).toBe(false);
  });
});
