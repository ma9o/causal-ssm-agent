import { describe, expect, it } from "vitest";
import { constructs, edges } from "../__fixtures__/dag-base-fixtures";
import { buildLayeredCausalGraph } from "./build-layered-causal-graph";
import { causalEdgeKey } from "./layered-causal-graph-model";

describe("buildLayeredCausalGraph", () => {
  it("keeps lagged, contemporaneous, and persistence topology distinct", () => {
    const built = buildLayeredCausalGraph({ constructs, edges });

    expect(
      built.edgeMeta.get(
        causalEdgeKey("internalizing_symptom_burden", "patient_taper_preference_beliefs", true),
      ),
    ).toMatchObject({
      source: "internalizing_symptom_burden__p",
      target: "patient_taper_preference_beliefs",
      lagged: true,
      isSelf: false,
    });
    expect(
      built.edgeMeta.get(
        causalEdgeKey("external_stressful_events", "perceived_stress_burden", false),
      ),
    ).toMatchObject({
      source: "external_stressful_events",
      target: "perceived_stress_burden",
      lagged: false,
      isSelf: false,
    });
    expect(built.edgeMeta.get("self:internalizing_symptom_burden")).toMatchObject({
      source: "internalizing_symptom_burden__p",
      target: "internalizing_symptom_burden",
      lagged: true,
      isSelf: true,
    });
  });

  it("builds topology only from structure", () => {
    const built = buildLayeredCausalGraph({ constructs, edges });
    expect([...built.nodeMeta.values()].filter((node) => node.kind === "construct")).toHaveLength(
      constructs.length,
    );
    expect([...built.edgeMeta.values()].filter((edge) => !edge.isSelf)).toHaveLength(edges.length);
  });
});
