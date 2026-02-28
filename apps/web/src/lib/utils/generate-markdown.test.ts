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
});
