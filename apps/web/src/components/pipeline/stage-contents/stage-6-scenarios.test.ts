import type {
  SimulateScenarioResult,
  Stage1aData,
  Stage4Data,
  Stage5bData,
} from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  counterfactualResult,
  interventionResult,
} from "@/components/dag/__fixtures__/intervention-dag-fixture";
import { materializedTrace } from "@/components/dag/__fixtures__/stage-6-materialized-fixture";
import stage1aFixture from "../../__fixtures__/demo-run/stage-1a.json";
import stage4Fixture from "../../__fixtures__/demo-run/stage-4.json";
import stage5bFixture from "../../__fixtures__/demo-run/stage-5b.json";
import { buildEdgePosteriors, buildStage6Scenarios } from "./stage-6-scenarios";

const stage1a = stage1aFixture as unknown as Stage1aData;
const stage4 = stage4Fixture as unknown as Stage4Data;
const stage5b = stage5bFixture as unknown as Stage5bData;

/** A refinement assistant turn carrying a live (object-valued) simulation result. */
function refinementSimMessage(
  toolCallId: string,
  result: SimulateScenarioResult,
  blurb = "Done.",
  input: unknown = {},
): UIMessage {
  return {
    id: `${toolCallId}-message`,
    role: "assistant",
    parts: [
      { type: "text", text: blurb },
      {
        type: "dynamic-tool",
        toolCallId,
        toolName: "simulate",
        state: "output-available",
        input,
        output: result,
      },
    ],
  };
}

describe("buildStage6Scenarios — baseline (no intervention)", () => {
  it("surfaces the clamp-less simulation as the single baseline, first", () => {
    const scenarios = buildStage6Scenarios({ trace: materializedTrace });

    const baselines = scenarios.filter((scenario) => scenario.provenance === "baseline");
    expect(baselines).toHaveLength(1);

    const baseline = scenarios[0];
    expect(baseline.provenance).toBe("baseline");
    expect(baseline.key).toBe("sim-0");
    expect(baseline.title).toBe("No intervention");
    expect(baseline.result.clamps).toHaveLength(0);
    // The LLM blurb produced with the scenario is carried through.
    expect(baseline.blurb).toContain("No intervention");
  });
});

describe("buildStage6Scenarios — interventions from a persisted trace", () => {
  it("recovers interventions from the trace where tool_result is a JSON string (reload path)", () => {
    const scenarios = buildStage6Scenarios({ trace: materializedTrace });

    // One baseline + four interventions (newest first).
    expect(scenarios).toHaveLength(5);
    expect(scenarios.slice(1).every((scenario) => scenario.provenance === "intervention")).toBe(
      true,
    );

    const newest = scenarios[1];
    expect(newest.key).toBe("sim-4");
    expect(newest.result.start.kind).toBe("baseline");
    expect(newest.title).toBe("do(serotonergic_exposure shift +1.0)");
    expect(newest.requestedHorizonDays).toBe(60);
    expect(newest.userQuery).toContain("serotonergic exposure by 1 SD");
    // The assistant text beside the tool call becomes the scenario blurb.
    expect(newest.blurb).toContain("Raising serotonergic exposure");
    // String-coerced result round-trips to the structured object.
    expect(newest.result.summary.mean).toBe(interventionResult.summary.mean);
    expect(newest.result.visualization?.node_effect_trajectories).toBeDefined();
  });

  it("captures abducted counterfactual fields and manifest projection", () => {
    const scenarios = buildStage6Scenarios({ trace: materializedTrace });

    const counterfactual = scenarios.find((scenario) => scenario.result.start.kind === "abducted");
    expect(counterfactual?.key).toBe("sim-3");
    expect(counterfactual?.result.summary.mean).toBe(counterfactualResult.summary.mean);

    // Manifest projection carried through on the set-mode simulation.
    const setMode = scenarios.find((scenario) => scenario.key === "sim-2");
    expect(setMode?.manifestEffects).toMatchObject({ state_of_mind_valence: 0.24 });
  });
});

describe("buildStage6Scenarios — trace ∪ extra messages", () => {
  it("dedupes by tool-call id with the extra-message copy winning and ranked newest", () => {
    const edited: SimulateScenarioResult = {
      ...interventionResult,
      summary: { ...interventionResult.summary, mean: 0.99 },
    };

    const scenarios = buildStage6Scenarios({
      trace: materializedTrace,
      extraMessages: [refinementSimMessage("sim-4", edited)],
    });

    // sim-4 is not duplicated…
    expect(scenarios).toHaveLength(5);
    expect(scenarios.filter((scenario) => scenario.key === "sim-4")).toHaveLength(1);
    // …the refinement copy wins and leads the interventions (after the baseline).
    expect(scenarios[1].key).toBe("sim-4");
    expect(scenarios[1].result.summary.mean).toBe(0.99);
  });

  it("places the baseline first, then interventions newest-first", () => {
    const scenarios = buildStage6Scenarios({ trace: materializedTrace });

    expect(scenarios[0].provenance).toBe("baseline");
    expect(scenarios.slice(1).map((scenario) => scenario.key)).toEqual([
      "sim-4",
      "sim-3",
      "sim-2",
      "sim-1",
    ]);
  });
});

describe("buildEdgePosteriors", () => {
  it("maps fixed-effect posterior marginals onto source→target edges", () => {
    const edgePosteriors = buildEdgePosteriors({
      stage1a: {
        latent_structure: {
          constructs: [
            {
              name: "stress_load",
              description: "Stress exposure",
              role: "endogenous",
              is_outcome: false,
              temporal_status: "time_varying",
            },
            {
              name: "sleep_quality",
              description: "Sleep quality",
              role: "endogenous",
              is_outcome: true,
              temporal_status: "time_varying",
            },
          ],
          edges: [
            {
              cause: "stress_load",
              effect: "sleep_quality",
              description: "Stress affects sleep",
              lagged: true,
            },
          ],
        },
      } as Stage1aData,
      stage4: {
        statistical_model_spec: {
          likelihoods: [],
          parameters: [
            {
              name: "beta_stress_load_sleep_quality",
              role: "fixed_effect",
              constraint: "none",
              description: "Effect of stress_load on sleep_quality",
            },
          ],
        },
        authored_priors: {},
        resolved_priors: [],
      } as unknown as Stage4Data,
      stage5b: {
        posterior_marginals: [
          {
            parameter: "beta_stress_load_sleep_quality",
            x_values: [0.1, 0.2],
            density: [1, 1],
            mean: 0.2,
            sd: 0.05,
            hdi_3: 0.1,
            hdi_97: 0.3,
          },
        ],
      } as unknown as Stage5bData,
    });

    expect(edgePosteriors).toEqual({
      "stress_load→sleep_quality": {
        mean: 0.2,
        ci_lower: 0.1,
        ci_upper: 0.3,
      },
    });
  });

  it("returns an empty map without stage 1a", () => {
    expect(buildEdgePosteriors({})).toEqual({});
  });
});
