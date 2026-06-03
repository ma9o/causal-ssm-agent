import type {
  SimulateScenarioResult,
  Stage1aData,
  Stage4Data,
  Stage5bData,
  TreatmentEffect,
} from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  counterfactualResult,
  interventionResult,
} from "@/components/dag/__fixtures__/intervention-dag-fixture";
import { materializedTrace } from "@/components/dag/__fixtures__/stage-6-materialized-fixture";
import stage1aFixture from "../../../../../../data/DEMO/run/stage-1a.json";
import stage4Fixture from "../../../../../../data/DEMO/run/stage-4.json";
import stage5bFixture from "../../../../../../data/DEMO/run/stage-5b.json";
import {
  buildEdgePosteriors,
  buildStage6Scenarios,
  type SimulationScenario,
} from "./stage-6-scenarios";

const stage1a = stage1aFixture as unknown as Stage1aData;
const stage4 = stage4Fixture as unknown as Stage4Data;
const stage5b = stage5bFixture as unknown as Stage5bData;

const OUTCOME = "affective_state";

const strongTreatment: TreatmentEffect = {
  treatment: "sleep_quality",
  posterior_draws: [0.1, 0.2, 0.3, 0.4, 0.5],
  temporal: {
    effect_1d: 0.05,
    effect_7d: 0.2,
    effect_30d: 0.3,
    peak_effect: 0.32,
    time_to_peak_days: 12,
  },
  manifest_effects: { state_of_mind_valence: 0.18 },
};

const weakTreatment: TreatmentEffect = {
  treatment: "seasonal_load",
  posterior_draws: [-0.05, -0.02, 0.0, 0.01, 0.03],
  temporal: null,
  manifest_effects: null,
};

const noDrawsTreatment: TreatmentEffect = {
  treatment: "ghost",
  posterior_draws: null,
  temporal: null,
  manifest_effects: null,
};

/** A refinement assistant turn carrying a live (object-valued) simulation result. */
function refinementSimMessage(
  toolCallId: string,
  result: SimulateScenarioResult,
  input: unknown = {},
): UIMessage {
  return {
    id: `${toolCallId}-message`,
    role: "assistant",
    parts: [
      { type: "text", text: "Done." },
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

describe("buildStage6Scenarios — baseline", () => {
  it("derives baseline scenarios from posterior draws and drops draw-less treatments", () => {
    const scenarios = buildStage6Scenarios({
      interventionResults: [weakTreatment, strongTreatment, noDrawsTreatment],
      outcomeName: OUTCOME,
    });

    expect(scenarios).toHaveLength(2);
    // Ordered by descending |mean|.
    expect(scenarios[0]).toMatchObject({
      provenance: "baseline",
      key: "baseline:sleep_quality",
      title: "sleep_quality",
      outcome: OUTCOME,
    });
    expect(scenarios[1].key).toBe("baseline:seasonal_load");

    const top = scenarios[0];
    expect(top.summary.mean).toBeCloseTo(0.3, 6);
    expect(top.summary.probPositive).toBeCloseTo(1, 6);
    expect(top.summary.lower95).toBeCloseTo(0.11, 6);
    expect(top.summary.upper95).toBeCloseTo(0.49, 6);
    expect(top.summary.peakEffect).toBe(0.32);
    expect(top.summary.timeToPeakDays).toBe(12);
    expect(top.manifestEffects).toEqual({ state_of_mind_valence: 0.18 });
    expect(scenarios[1].manifestEffects).toBeNull();
  });
});

describe("buildStage6Scenarios — simulations from a persisted trace", () => {
  it("recovers simulations from the trace where tool_result is a JSON string (reload path)", () => {
    const scenarios = buildStage6Scenarios({
      interventionResults: [],
      outcomeName: OUTCOME,
      trace: materializedTrace,
    });

    // Five synthesized simulations, newest first.
    expect(scenarios).toHaveLength(5);
    expect(scenarios.every((scenario) => scenario.provenance === "simulation")).toBe(true);

    const newest = scenarios[0] as SimulationScenario;
    expect(newest.key).toBe("sim-4");
    expect(newest.result.start.kind).toBe("baseline");
    expect(newest.title).toBe("do(serotonergic_exposure shift +1.0)");
    expect(newest.requestedHorizonDays).toBe(60);
    expect(newest.userQuery).toContain("serotonergic exposure by 1 SD");
    // String-coerced result round-trips to the structured object.
    expect(newest.result.summary.mean).toBe(interventionResult.summary.mean);
    expect(newest.result.visualization?.node_effect_trajectories).toBeDefined();
  });

  it("captures abducted counterfactual fields and the end-state degraded path", () => {
    const scenarios = buildStage6Scenarios({
      interventionResults: [],
      outcomeName: OUTCOME,
      trace: materializedTrace,
    }) as SimulationScenario[];

    const counterfactual = scenarios.find((scenario) => scenario.result.start.kind === "abducted");
    expect(counterfactual?.key).toBe("sim-3");
    expect(counterfactual?.result.summary.mean).toBe(counterfactualResult.summary.mean);

    // The steady_state simulation has no trajectory → no derivable peak.
    const steadyState = scenarios.find((scenario) => scenario.key === "sim-1");
    expect(steadyState?.result.effect_trajectory ?? null).toBeNull();
    expect(steadyState?.summary.peakEffect).toBeNull();
    expect(steadyState?.summary.timeToPeakDays).toBeNull();

    // Manifest projection carried through on the set-mode simulation.
    const setMode = scenarios.find((scenario) => scenario.key === "sim-2");
    expect(setMode?.manifestEffects).toMatchObject({ state_of_mind_valence: 0.24 });
  });
});

describe("buildStage6Scenarios — trace ∪ refinement", () => {
  it("dedupes by tool-call id with the live refinement copy winning and ranked newest", () => {
    const edited: SimulateScenarioResult = {
      ...interventionResult,
      summary: { ...interventionResult.summary, mean: 0.99 },
    };

    const scenarios = buildStage6Scenarios({
      interventionResults: [],
      outcomeName: OUTCOME,
      trace: materializedTrace,
      refinementMessages: [refinementSimMessage("sim-4", edited)],
    }) as SimulationScenario[];

    // sim-4 is not duplicated…
    expect(scenarios).toHaveLength(5);
    expect(scenarios.filter((scenario) => scenario.key === "sim-4")).toHaveLength(1);
    // …the refinement copy wins…
    expect(scenarios[0].key).toBe("sim-4");
    expect(scenarios[0].result.summary.mean).toBe(0.99);
  });

  it("places simulations before baselines (sims newest-first, baselines by |effect|)", () => {
    const scenarios = buildStage6Scenarios({
      interventionResults: [weakTreatment, strongTreatment],
      outcomeName: OUTCOME,
      trace: materializedTrace,
    });

    expect(scenarios).toHaveLength(7);
    expect(scenarios.slice(0, 5).every((scenario) => scenario.provenance === "simulation")).toBe(
      true,
    );
    expect(scenarios[5]).toMatchObject({ provenance: "baseline", key: "baseline:sleep_quality" });
    expect(scenarios[6].key).toBe("baseline:seasonal_load");
  });
});

describe("buildEdgePosteriors", () => {
  it("maps fixed-effect posterior marginals onto source→target edges", () => {
    const edgePosteriors = buildEdgePosteriors({ stage1a, stage4, stage5b });
    expect(Object.keys(edgePosteriors).length).toBeGreaterThan(0);
    for (const posterior of Object.values(edgePosteriors)) {
      expect(posterior).toHaveProperty("mean");
      expect(posterior).toHaveProperty("ci_lower");
      expect(posterior).toHaveProperty("ci_upper");
    }
  });

  it("returns an empty map without stage 1a", () => {
    expect(buildEdgePosteriors({})).toEqual({});
  });
});
