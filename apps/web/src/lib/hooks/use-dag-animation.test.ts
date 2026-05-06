import { describe, expect, it } from "vitest";
import type { Stage1aData } from "@causal-ssm/api-types";
import stage1aFixture from "../../../../../data/DEMO_HEALTH/run/stage-1a.json";
import {
  counterfactualResult,
  interventionResult,
} from "@/components/dag/__fixtures__/intervention-dag-fixture";
import { deriveDagAnimationFrame } from "./use-dag-animation";

const stage1a = stage1aFixture as unknown as Stage1aData;
const constructs = stage1a.latent_model.constructs;
const edges = stage1a.latent_model.edges;

describe("deriveDagAnimationFrame", () => {
  it("dims nodes and edges outside the causal cone for rung 2", () => {
    const frame = deriveDagAnimationFrame(0.5, {
      constructs,
      edges,
      result: interventionResult,
    });

    expect(frame?.phase).toBe("propagating");
    expect(frame?.nodePhases.vascular_inflammation).toBe("active");
    expect(frame?.nodePhases.glycemic_control).toBe("dimmed");
    expect(frame?.edgeStates["psychosocial_stress→glycemic_control"]).toBe("dimmed");
    expect(frame?.edgeStates["genetic_predisposition→lipid_burden"]).toBe("cut");
  });

  it("keeps downstream zero-effect nodes visible and animated", () => {
    const result = structuredClone(interventionResult);
    if (result.visualization?.node_effect_trajectories != null) {
      result.visualization.node_effect_trajectories.vascular_inflammation =
        result.visualization.node_effect_trajectories.vascular_inflammation.map(() => 0);
    }

    const frame = deriveDagAnimationFrame(0.5, {
      constructs,
      edges,
      result,
    });

    expect(frame?.nodePhases.vascular_inflammation).toBe("active");
    expect(frame?.nodePhases.glycemic_control).toBe("dimmed");
    expect(frame?.edgeStates["lipid_burden→vascular_inflammation"]).toBe("flowing");
  });

  it("uses the same structural dimming rule during rung-3 prediction", () => {
    const frame = deriveDagAnimationFrame(0.6, {
      constructs,
      edges,
      result: counterfactualResult,
    });

    expect(frame?.phase).toBe("prediction");
    expect(frame?.nodePhases.lipid_burden).toBe("active");
    expect(frame?.nodePhases.genetic_predisposition).toBe("dimmed");
    expect(frame?.nodePhases.psychosocial_stress).toBe("dimmed");
    expect(frame?.edgeStates["genetic_predisposition→lipid_burden"]).toBe("dimmed");
  });
});
