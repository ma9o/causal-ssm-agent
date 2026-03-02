import { describe, expect, it } from "vitest";

// The helper functions are not exported, so we test them through the module.
// We can test the exported generateMarkdown with minimal stage data.
import { type AllStageData, generateMarkdown } from "./generate-markdown";

describe("generateMarkdown", () => {
  it("generates header with run ID", () => {
    const data: AllStageData = {};
    const result = generateMarkdown(data, "test-run-123");
    expect(result).toContain("# Causal Inference Pipeline Report");
    expect(result).toContain("`test-run-123`");
    expect(result).toContain("**Generated**:");
  });

  it("generates empty report without crashing", () => {
    const data: AllStageData = {};
    const result = generateMarkdown(data, "empty");
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  it("includes stage 0 section when data is present", () => {
    const data: AllStageData = {
      "stage-0": {
        outcome: "success",
        source_type: "csv",
        source_label: "test_data.csv",
        n_records: 100,
        date_range: { start: "2024-01-01", end: "2024-12-31" },
        sample: [{ timestamp: "2024-01-01", value: "42" }],
      },
    };
    const result = generateMarkdown(data, "run-1");
    expect(result).toContain("Stage 0");
    expect(result).toContain("100");
  });

  it("handles null stage data gracefully", () => {
    const data: AllStageData = {
      "stage-0": null,
      "stage-1a": null,
    };
    const result = generateMarkdown(data, "null-test");
    expect(typeof result).toBe("string");
  });

  it("includes stage 6 treatment effects section", () => {
    const data: AllStageData = {
      "stage-6": {
        outcome: "success",
        intervention_results: [
          {
            treatment: "exercise",
            effect_size: 0.35,
            identifiable: true,
            prob_positive: 0.92,
            posterior_draws: [0.1, 0.2, 0.3, 0.4, 0.5],
          },
        ],
        inference_metadata: {
          method: "svi",
          n_samples: 1000,
          duration_seconds: 30.5,
        },
      } as AllStageData["stage-6"],
    };
    const result = generateMarkdown(data, "run-effects");
    expect(result).toContain("exercise");
    expect(result).toContain("Treatment");
  });

  it("includes multiple stages together", () => {
    const data: AllStageData = {
      "stage-0": {
        outcome: "success",
        source_type: "csv",
        source_label: "data.csv",
        n_records: 50,
        date_range: { start: "2024-01-01", end: "2024-06-30" },
        sample: [],
      },
      "stage-3": {
        outcome: "success",
        validation_report: {
          is_valid: true,
          issues: [],
          per_indicator_health: [],
        },
      } as AllStageData["stage-3"],
    };
    const result = generateMarkdown(data, "multi-stage");
    expect(result).toContain("Stage 0");
    expect(result).toContain("Stage 3");
  });

  it("contains date in header", () => {
    const result = generateMarkdown({}, "test");
    // Should have ISO-like date format
    expect(result).toMatch(/\d{4}/);
  });

  it("includes stage 1a constructs and edges", () => {
    const data: AllStageData = {
      "stage-1a": {
        outcome: "success",
        outcome_name: "blood_pressure",
        treatments: ["exercise", "diet"],
        latent_model: {
          constructs: [
            {
              name: "exercise",
              description: "Physical activity",
              role: "exogenous",
              is_outcome: false,
              temporal_status: "time_varying",
            },
            {
              name: "blood_pressure",
              description: "BP measurement",
              role: "endogenous",
              is_outcome: true,
              temporal_status: "time_varying",
            },
          ],
          edges: [
            {
              cause: "exercise",
              effect: "blood_pressure",
              lagged: true,
              description: "Regular exercise lowers BP",
            },
          ],
        },
      } as AllStageData["stage-1a"],
    };
    const result = generateMarkdown(data, "run-1a");
    expect(result).toContain("exercise");
    expect(result).toContain("blood_pressure");
    expect(result).toContain("Stage 1a");
    expect(result).toContain("Constructs");
  });

  it("includes stage 3 validation issues", () => {
    const data: AllStageData = {
      "stage-3": {
        outcome: "success",
        validation_report: {
          is_valid: false,
          issues: [
            { type: "error", message: "Missing data in column X" },
            { type: "warning", message: "Low variance in Y" },
          ],
          per_indicator_health: [
            {
              indicator: "heart_rate",
              n_obs: 100,
              n_missing: 5,
              mean: 72.5,
              std: 10.2,
              min: 55,
              max: 120,
            },
          ],
        },
      } as AllStageData["stage-3"],
    };
    const result = generateMarkdown(data, "run-3");
    expect(result).toContain("Stage 3");
    expect(result).toContain("heart_rate");
  });

  it("handles stage 5 MCMC diagnostics", () => {
    const data: AllStageData = {
      "stage-5": {
        outcome: "success",
        inference_metadata: {
          method: "nuts",
          n_samples: 2000,
          duration_seconds: 120,
        },
        mcmc_diagnostics: {
          num_divergences: 0,
          divergence_rate: 0,
          tree_depth_mean: 5.2,
          tree_depth_max: 8,
          accept_prob_mean: 0.85,
          num_chains: 4,
          num_samples: 2000,
          per_parameter: [
            {
              parameter: "drift_diag_0",
              r_hat: 1.001,
              ess_bulk: 1800,
              ess_tail: 1600,
            },
          ],
        },
        ppc: {
          per_variable_warnings: [
            {
              variable: "heart_rate",
              check_type: "mean",
              passed: true,
              value: 0.12,
              message: "OK",
            },
          ],
        },
        power_scaling: [],
      } as AllStageData["stage-5"],
    };
    const result = generateMarkdown(data, "run-5");
    expect(result).toContain("Stage 5");
    expect(result).toContain("MCMC Diagnostics");
    expect(result).toContain("drift_diag_0");
  });
});
