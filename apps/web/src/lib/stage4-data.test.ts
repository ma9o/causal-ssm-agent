import type { Stage4Data } from "@causal-ssm/api-types";
import { describe, expect, it } from "vitest";

import { collectStage4UiPriors } from "./stage4-data";

const stage4Data = {
  outcome: "success",
  model_spec: {
    likelihoods: [],
    parameters: [
      {
        name: "rho_sleep",
        role: "ar_coefficient",
        constraint: "unit_interval",
        description: "Persistence for sleep.",
      },
      {
        name: "beta_stress_sleep",
        role: "fixed_effect",
        constraint: "none",
        description: "Effect of stress on sleep.",
      },
      {
        name: "sigma_sleep",
        role: "residual_sd",
        constraint: "positive",
        description: "Innovation scale for sleep.",
      },
    ],
    initialization_policy: "free",
    equilibrium_forcing: false,
  },
  authored_priors: {
    rho_sleep: {
      parameter: "rho_sleep",
      distribution: "Beta",
      params: { alpha: 3, beta: 2 },
      sources: [],
      reasoning: "Daily persistence prior.",
    },
    beta_stress_sleep: {
      parameter: "beta_stress_sleep",
      distribution: "Normal",
      params: { mu: -0.2, sigma: 0.1 },
      sources: [],
      reasoning: "Lagged effect prior.",
    },
    orphan_prior: {
      parameter: "orphan_prior",
      distribution: "Normal",
      params: { mu: 0, sigma: 1 },
      sources: [],
      reasoning: "Should not be shown in the semantic UI.",
    },
  },
  resolved_priors: [
    {
      parameter: "rho_sleep",
      distribution: "Beta",
      params: { alpha: 9, beta: 1 },
      sources: [],
      reasoning: "Compiled version that should stay hidden in the UI.",
    },
    {
      parameter: "sigma_sleep",
      distribution: "HalfNormal",
      params: { sigma: 0.4 },
      sources: [],
      reasoning: "Implicit compiler default that should stay hidden in the UI.",
    },
  ],
} as Stage4Data;

describe("collectStage4UiPriors", () => {
  it("returns only authored priors that correspond to declared model parameters", () => {
    expect(collectStage4UiPriors(stage4Data).map((prior) => prior.parameter)).toEqual([
      "rho_sleep",
      "beta_stress_sleep",
    ]);
  });
});
