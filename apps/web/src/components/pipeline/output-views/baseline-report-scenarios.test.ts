import type {
  SimulateScenarioResult,
  LatentStructureData,
  StatisticalModelSpecData,
  PosteriorData,
} from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import { demoBaselineTrace } from "@/components/dag/__fixtures__/baseline_report-materialized-fixture";
import {
  buildBaselineReportScenarios,
  buildEdgePosteriors,
  buildPersistencePosteriors,
} from "./baseline-report-scenarios";

const fixtureScenarios = buildBaselineReportScenarios({ trace: demoBaselineTrace });
const interventionResult = fixtureScenarios.find((scenario) => scenario.key === "sim-5")?.result;
const counterfactualResult = fixtureScenarios.find((scenario) => scenario.key === "sim-4")?.result;

if (!interventionResult || !counterfactualResult) {
  throw new Error("The canonical DEMO trace is missing its materialized test scenarios.");
}

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

describe("buildBaselineReportScenarios — interventions from a persisted trace", () => {
  it("recovers interventions from the trace where tool_result is a JSON string (reload path)", () => {
    const scenarios = buildBaselineReportScenarios({ trace: demoBaselineTrace });

    expect(scenarios).toHaveLength(5);
    expect(scenarios.every((scenario) => scenario.provenance === "intervention")).toBe(true);

    const newest = scenarios[0];
    expect(newest.key).toBe("sim-5");
    expect(newest.result.start.kind).toBe("baseline");
    expect(newest.title).toBe("do(taper_speed_dose_reduction set 0.9)");
    expect(newest.requestedHorizonDays).toBe(60);
    expect(newest.userQuery).toContain("taper speed is raised sharply");
    // The assistant text beside the tool call becomes the scenario blurb.
    expect(newest.blurb).toContain("Rapid taper");
    // String-coerced result round-trips to the structured object.
    expect(newest.result.summary.mean).toBe(interventionResult.summary.mean);
    expect(newest.result.visualization?.node_effect_trajectories).toBeDefined();
  });

  it("captures abducted counterfactual fields and manifest projection", () => {
    const scenarios = buildBaselineReportScenarios({ trace: demoBaselineTrace });

    const counterfactual = scenarios.find((scenario) => scenario.result.start.kind === "abducted");
    expect(counterfactual?.key).toBe("sim-4");
    expect(counterfactual?.result.summary.mean).toBe(counterfactualResult.summary.mean);

    // Manifest projection carried through on the set-mode simulation.
    const setMode = scenarios.find((scenario) => scenario.key === "sim-5");
    expect(setMode?.manifestEffects).toHaveProperty("state_of_mind_valence");
  });
});

describe("buildBaselineReportScenarios — trace ∪ extra messages", () => {
  it("dedupes by tool-call id with the extra-message copy winning and ranked newest", () => {
    const edited: SimulateScenarioResult = {
      ...interventionResult,
      summary: { ...interventionResult.summary, mean: 0.99 },
    };

    const scenarios = buildBaselineReportScenarios({
      trace: demoBaselineTrace,
      extraMessages: [refinementSimMessage("sim-5", edited)],
    });

    // sim-5 is not duplicated…
    expect(scenarios).toHaveLength(5);
    expect(scenarios.filter((scenario) => scenario.key === "sim-5")).toHaveLength(1);
    // …the refinement copy wins and leads the interventions.
    expect(scenarios[0].key).toBe("sim-5");
    expect(scenarios[0].result.summary.mean).toBe(0.99);
  });

  it("orders production-valid interventions newest-first", () => {
    const scenarios = buildBaselineReportScenarios({ trace: demoBaselineTrace });

    expect(scenarios.map((scenario) => scenario.key)).toEqual([
      "sim-5",
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
      latentStructure: {
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
      } as LatentStructureData,
      modelSpec: {
        statistical_model_spec: {
          likelihoods: [],
          parameters: [
            {
              name: "coefficient_42",
              role: "fixed_effect",
              constraint: "none",
              description: "Effect of stress_load on sleep_quality",
            },
          ],
        },
        authored_priors: {},
        resolved_priors: [],
      } as unknown as StatisticalModelSpecData,
      posterior: {
        posterior_marginals: [
          {
            parameter: "coefficient_42",
            x_values: [0.1, 0.2],
            density: [1, 1],
            mean: 0.2,
            sd: 0.05,
            hdi_3: 0.1,
            hdi_97: 0.3,
          },
        ],
      } as unknown as PosteriorData,
    });

    expect(edgePosteriors).toEqual({
      "stress_load→sleep_quality": {
        mean: 0.2,
        ci_lower: 0.1,
        ci_upper: 0.3,
      },
    });
  });

  it("returns an empty map without latent_structure", () => {
    expect(buildEdgePosteriors({})).toEqual({});
  });
});

describe("buildPersistencePosteriors", () => {
  it("maps only backend-declared AR parameters onto fitted latent states", () => {
    const persistence = buildPersistencePosteriors({
      modelSpec: {
        statistical_model_spec: {
          likelihoods: [],
          parameters: [
            {
              name: "rho_sleep_quality",
              role: "ar_coefficient",
              constraint: "unit_interval",
              description: "Baseline daily persistence absent incoming feedback for sleep_quality",
            },
            {
              name: "sigma_sleep_quality",
              role: "residual_sd",
              constraint: "positive",
              description: "Residual scale for sleep_quality",
            },
          ],
        },
        authored_priors: {},
        resolved_priors: [],
      } as unknown as StatisticalModelSpecData,
      posterior: {
        posterior_marginals: [
          {
            parameter: "rho_sleep_quality",
            x_values: [0.7, 0.8],
            density: [1, 1],
            mean: 0.76,
            sd: 0.03,
            hdi_3: 0.7,
            hdi_97: 0.82,
          },
          {
            parameter: "sigma_sleep_quality",
            x_values: [0.1, 0.2],
            density: [1, 1],
            mean: 0.15,
            sd: 0.02,
            hdi_3: 0.11,
            hdi_97: 0.19,
          },
        ],
      } as unknown as PosteriorData,
    });

    expect(persistence).toEqual({
      sleep_quality: { mean: 0.76, ci_lower: 0.7, ci_upper: 0.82 },
    });
  });
});
