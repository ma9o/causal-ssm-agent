import { describe, expect, it } from "vitest";
import { design, structuralPlan } from "../__fixtures__/dag-base-fixtures";
import {
  availableGraphLayers,
  causalEdgeKey,
  deriveEdgeDesignDispositions,
  type LayeredCausalGraphModel,
} from "./layered-causal-graph-model";

describe("layered causal graph model", () => {
  it("exposes only cumulatively materialized layers", () => {
    const structure: LayeredCausalGraphModel = { structure: design.latent };
    expect(availableGraphLayers(structure)).toEqual(["structure"]);

    const measurement: LayeredCausalGraphModel = {
      ...structure,
      measurement: {
        measurement: design.measurement,
        knownInputs: design.known_inputs,
        scientificOnlyConstructs: design.scientific_only_constructs,
      },
    };
    expect(availableGraphLayers(measurement)).toEqual(["structure", "measurement"]);
  });

  it("rejects a later layer when its immediate dependency is absent", () => {
    expect(() =>
      availableGraphLayers({
        structure: design.latent,
        design: { causalDesign: design, structuralPlan },
      }),
    ).toThrow("requires 'measurement'");
  });

  it("maps every backend edge disposition by timing-aware semantic identity", () => {
    const dispositions = deriveEdgeDesignDispositions(structuralPlan);
    expect(dispositions.size).toBe(design.latent.edges.length);
    for (const edge of design.latent.edges) {
      expect(dispositions.get(causalEdgeKey(edge.cause, edge.effect, edge.lagged))).toMatch(
        /^(retained_edge|projected_edge)$/,
      );
    }
  });
});
