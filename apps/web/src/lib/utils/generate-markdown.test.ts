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
});
