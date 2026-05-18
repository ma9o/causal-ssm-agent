import { describe, expect, it } from "vitest";
import type { IndicatorAudit, LikelihoodSpec, ObservationRecord } from "@nof1-causal-lab/api-types";
import { buildStage4LikelihoodDiagnostics } from "./stage4-likelihood-diagnostics";

describe("buildStage4LikelihoodDiagnostics", () => {
  it("builds per-likelihood histograms from full observations and preserves stage-3 profiles", () => {
    const likelihoods: LikelihoodSpec[] = [
      {
        variable: "screen_time_count",
        distribution: "negative_binomial",
        link: "log",
        centered: false,
        reasoning: "Count data",
        sources: [],
      },
      {
        variable: "sleep_latency",
        distribution: "gaussian",
        link: "identity",
        centered: false,
        reasoning: "Continuous data",
        sources: [],
      },
    ];

    const screenProfile = {
      measurement_dtype: "count",
      n_obs: 3,
      mean: 1.67,
      std: 0.58,
      min: 1,
      max: 2,
      q25: 1,
      q50: 2,
      q75: 2,
      variance: 0.33,
      time_coverage_ratio: 1,
      max_gap_ratio: 0.2,
      dtype_violations: 0,
      duplicate_pct: 0.67,
      arithmetic_sequence_detected: false,
      n_unparseable_timestamps: 0,
      zero_fraction: 0,
      is_nonnegative: true,
      is_unit_interval: false,
      looks_integer_valued: true,
      variance_to_mean_ratio: 0.2,
    } as const;
    const sleepProfile = {
      measurement_dtype: "continuous",
      n_obs: 3,
      mean: 12,
      std: 2,
      min: 10,
      max: 14,
      q25: 11,
      q50: 12,
      q75: 13,
      variance: 4,
      time_coverage_ratio: 1,
      max_gap_ratio: 0.2,
      dtype_violations: 0,
      duplicate_pct: 0.33,
      arithmetic_sequence_detected: false,
      n_unparseable_timestamps: 0,
      zero_fraction: 0,
      is_nonnegative: true,
      is_unit_interval: false,
      looks_integer_valued: true,
      variance_to_mean_ratio: 0.33,
    } as const;

    const indicatorAudits: Record<string, IndicatorAudit> = {
      screen_time_count: { profile: screenProfile, validation: { issues: [], checks: {} } },
      sleep_latency: { profile: sleepProfile, validation: { issues: [], checks: {} } },
    };

    const observations: ObservationRecord[] = [
      { indicator: "screen_time_count", value: "1", anchor_time: null },
      { indicator: "screen_time_count", value: 2, anchor_time: null },
      { indicator: "screen_time_count", value: 2, anchor_time: null },
      { indicator: "screen_time_count", value: null, anchor_time: null },
      { indicator: "sleep_latency", value: 10, anchor_time: null },
      { indicator: "sleep_latency", value: 12, anchor_time: null },
      { indicator: "sleep_latency", value: 14, anchor_time: null },
      { indicator: "sleep_latency", value: "not-a-number", anchor_time: null },
      { indicator: "other_indicator", value: 99, anchor_time: null },
    ];

    const diagnostics = buildStage4LikelihoodDiagnostics({
      likelihoods,
      indicatorAudits,
      observations,
    });

    expect(diagnostics.screen_time_count?.profile).toEqual(screenProfile);
    expect(diagnostics.screen_time_count?.histogram).toEqual([
      { binCenter: 1, count: 1 },
      { binCenter: 2, count: 2 },
    ]);

    expect(diagnostics.sleep_latency?.profile).toEqual(sleepProfile);
    expect(diagnostics.sleep_latency?.histogram.length).toBeGreaterThan(0);
    expect(diagnostics.sleep_latency?.histogram.reduce((sum, bin) => sum + bin.count, 0)).toBe(3);
  });
});
