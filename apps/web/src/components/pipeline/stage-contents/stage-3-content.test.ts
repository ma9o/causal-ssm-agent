import { describe, expect, it } from "vitest";
import type { Stage3Data } from "@nof1-causal-lab/api-types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { buildFixPrompt, getFixPromptData, Stage3FixAction } from "./stage-3-content";

function makeStage3Data(overrides: Partial<Stage3Data> = {}): Stage3Data {
  return {
    outcome: "warn",
    is_valid: false,
    dataset_issues: [],
    indicators: {},
    ...overrides,
  };
}

describe("buildFixPrompt", () => {
  it("returns a repair prompt for warning-only validation issues", () => {
    const data = makeStage3Data({
      indicators: {
        sleep_quality: {
          profile: {
            measurement_dtype: null,
            n_obs: 12,
            mean: null,
            std: null,
            min: null,
            max: null,
            q25: null,
            q50: null,
            q75: null,
            variance: null,
            time_coverage_ratio: null,
            max_gap_ratio: null,
            dtype_violations: 0,
            duplicate_pct: 0,
            arithmetic_sequence_detected: false,
            n_unparseable_timestamps: null,
            zero_fraction: null,
            is_nonnegative: null,
            is_unit_interval: null,
            looks_integer_valued: null,
            variance_to_mean_ratio: null,
          },
          validation: {
            issues: [
              {
                indicator: "sleep_quality",
                issue_type: "low_n",
                severity: "warning",
                message: "Only 12 observations remain after filtering.",
              },
            ],
            checks: {},
          },
        },
      },
    });
    const { prompt, highestSeverity } = getFixPromptData(data);

    expect(prompt).toContain("Stage 3 (Validation) surfaced");
    expect(prompt).toContain("Warnings");
    expect(prompt).toContain("sleep_quality: Only 12 observations remain after filtering.");
    expect(highestSeverity).toBe("warning");
  });

  it("prefers error when both warnings and errors are present", () => {
    const { highestSeverity } = getFixPromptData(
      makeStage3Data({
        dataset_issues: [
          {
            issue_type: "low_n",
            severity: "warning",
            message: "Warning issue.",
          },
          {
            issue_type: "missingness",
            severity: "error",
            message: "Error issue.",
          },
        ],
      }),
    );

    expect(highestSeverity).toBe("error");
  });

  it("returns an empty prompt when validation only has info-level issues", () => {
    const prompt = buildFixPrompt(
      makeStage3Data({
        dataset_issues: [
          {
            issue_type: "note",
            severity: "info",
            message: "Informational only.",
          },
        ],
      }),
    );

    expect(prompt).toBe("");
  });

  it("renders a warning-styled repair action for warning-only issues", () => {
    const markup = renderToStaticMarkup(
      createElement(Stage3FixAction, {
        data: makeStage3Data({
          dataset_issues: [
            {
              issue_type: "low_n",
              severity: "warning",
              message: "Warning issue.",
            },
          ],
        }),
        onFix: () => undefined,
      }),
    );

    expect(markup).toContain("Propose fixes");
    expect(markup).toContain("bg-warning/15");
    expect(markup).not.toContain("bg-destructive/10");
  });
});
