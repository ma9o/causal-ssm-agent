import { describe, expect, it } from "vitest";
import type { Stage1aData } from "@nof1-causal-lab/api-types";
import stage1aFixture from "../../../../../data/DEMO/run/stage-1a.json";
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
    expect(frame?.nodePhases.affective_state).toBe("active");
    expect(frame?.nodePhases.seasonal_load).toBe("dimmed");
    expect(frame?.edgeStates["seasonal_load→affective_state"]).toBe("dimmed");
    expect(frame?.edgeStates["cyp2c19_metabolizer_status→serotonergic_exposure"]).toBe("cut");
  });

  it("keeps downstream zero-effect nodes visible and animated", () => {
    const result = structuredClone(interventionResult);
    if (result.visualization?.node_effect_trajectories != null) {
      const affectiveStateEffects = result.visualization.node_effect_trajectories.affective_state;
      if (affectiveStateEffects) {
        result.visualization.node_effect_trajectories.affective_state = affectiveStateEffects.map(
          () => 0,
        );
      }
    }

    const frame = deriveDagAnimationFrame(0.5, {
      constructs,
      edges,
      result,
    });

    expect(frame?.nodePhases.affective_state).toBe("active");
    expect(frame?.nodePhases.seasonal_load).toBe("dimmed");
    expect(frame?.edgeStates["serotonergic_exposure→affective_state"]).toBe("flowing");
  });

  it("uses the same structural dimming rule during rung-3 prediction", () => {
    const frame = deriveDagAnimationFrame(0.6, {
      constructs,
      edges,
      result: counterfactualResult,
    });

    expect(frame?.phase).toBe("prediction");
    expect(frame?.nodePhases.serotonergic_exposure).toBe("active");
    expect(frame?.nodePhases.cyp2c19_metabolizer_status).toBe("dimmed");
    expect(frame?.nodePhases.life_events_load).toBe("dimmed");
    expect(frame?.edgeStates["cyp2c19_metabolizer_status→serotonergic_exposure"]).toBe("dimmed");
  });
});
