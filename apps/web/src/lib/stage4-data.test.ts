import type { Indicator, Stage4Data } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";

import { collectStage4ObservationPriorTerms, collectStage4UiPriors } from "./stage4-data";

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
  likelihood_diagnostics: {},
} as Stage4Data;

function makeIndicator(overrides: Partial<Indicator> = {}): Indicator {
  return {
    name: "mood",
    construct_name: "affect",
    how_to_measure: "Extract mood ratings.",
    construct_polarity: "positive",
    measurement_dtype: "continuous",
    aggregation: "mean",
    source_columns: [],
    extraction_mode: "semantic",
    support_kind: "point",
    summary_operator: "mean",
    anchor_policy: "support_end",
    ...overrides,
  };
}

describe("collectStage4UiPriors", () => {
  it("returns only authored priors that correspond to declared model parameters", () => {
    expect(collectStage4UiPriors(stage4Data).map((prior) => prior.parameter)).toEqual([
      "rho_sleep",
      "beta_stress_sleep",
    ]);
  });
});

describe("collectStage4ObservationPriorTerms", () => {
  it("maps measurement-error and family-specific observation priors to one likelihood row", () => {
    const terms = collectStage4ObservationPriorTerms({
      likelihood: {
        variable: "appointment_attendance",
        distribution: "beta",
        link: "logit",
        centered: false,
        reasoning: "",
        sources: [],
      },
      parameters: [
        {
          name: "lambda_appointment_attendance_medication_adherence",
          role: "loading",
          constraint: "positive",
          description: "Loading for appointment attendance.",
        },
        {
          name: "obs_sd_appointment_attendance",
          role: "measurement_error_sd",
          constraint: "positive",
          description: "Measurement-error SD for appointment attendance.",
        },
        {
          name: "obs_concentration",
          role: "observation_hyperparameter_positive",
          constraint: "positive",
          description: "Beta observation concentration.",
        },
      ],
      priors: [
        {
          parameter: "lambda_appointment_attendance_medication_adherence",
          distribution: "HalfNormal",
          params: { sigma: 1 },
          sources: [],
          reasoning: "",
        },
        {
          parameter: "obs_sd_appointment_attendance",
          distribution: "HalfNormal",
          params: { sigma: 0.5 },
          sources: [],
          reasoning: "",
        },
        {
          parameter: "obs_concentration",
          distribution: "Gamma",
          params: { alpha: 5, beta: 0.5 },
          sources: [],
          reasoning: "",
        },
      ],
      indicators: [
        makeIndicator({
          name: "appointment_attendance",
          construct_name: "medication_adherence",
        }),
      ],
    });

    expect(terms.map((term) => term.parameterName)).toEqual([
      "lambda_appointment_attendance_medication_adherence",
      "obs_sd_appointment_attendance",
      "obs_concentration",
    ]);
    expect(terms.every((term) => term.prior)).toBe(true);
  });

  it("only includes ordered threshold gaps when the indicator has more than two levels", () => {
    const terms = collectStage4ObservationPriorTerms({
      likelihood: {
        variable: "stress_level",
        distribution: "ordered_logistic",
        link: "cumulative_logit",
        centered: false,
        reasoning: "",
        sources: [],
      },
      parameters: [
        {
          name: "obs_ordered_base",
          role: "observation_hyperparameter",
          constraint: "none",
          description: "Ordered threshold bases.",
        },
        {
          name: "obs_ordered_gaps",
          role: "observation_hyperparameter_positive",
          constraint: "positive",
          description: "Ordered threshold gaps.",
        },
      ],
      priors: [
        {
          parameter: "obs_ordered_base",
          distribution: "Normal",
          params: { mu: 0, sigma: 1 },
          sources: [],
          reasoning: "",
        },
        {
          parameter: "obs_ordered_gaps",
          distribution: "HalfNormal",
          params: { sigma: 1 },
          sources: [],
          reasoning: "",
        },
      ],
      indicators: [
        makeIndicator({
          name: "stress_level",
          measurement_dtype: "ordinal",
          summary_operator: "last",
          ordinal_levels: ["low", "high"],
        }),
      ],
    });

    expect(terms.map((term) => term.parameterName)).toEqual(["obs_ordered_base"]);
  });
});
