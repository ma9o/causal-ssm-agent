import { describe, expect, it } from "vitest";
import {
  CI_LOWER,
  CI_UPPER,
  DEFAULT_N_SAMPLES,
  ESS_RATIO_FAIL,
  ESS_RATIO_WARN,
  PARETO_K_FAIL,
  PARETO_K_WARN,
  POWER_SCALING_THRESHOLD,
  PPC_P_LOWER,
  PPC_P_UPPER,
  RHAT_FAIL,
  RHAT_WARN,
} from "./diagnostics";

describe("diagnostics constants", () => {
  it("R-hat thresholds are ordered correctly", () => {
    expect(RHAT_WARN).toBeLessThan(RHAT_FAIL);
    expect(RHAT_WARN).toBeGreaterThan(1);
  });

  it("ESS ratio thresholds are ordered correctly", () => {
    expect(ESS_RATIO_FAIL).toBeLessThan(ESS_RATIO_WARN);
    expect(ESS_RATIO_FAIL).toBeGreaterThan(0);
    expect(ESS_RATIO_WARN).toBeLessThanOrEqual(1);
  });

  it("Pareto-k thresholds are ordered correctly", () => {
    expect(PARETO_K_WARN).toBeLessThan(PARETO_K_FAIL);
    expect(PARETO_K_WARN).toBeGreaterThan(0);
  });

  it("PPC p-value thresholds are symmetric around 0.5", () => {
    expect(PPC_P_LOWER + PPC_P_UPPER).toBe(1);
    expect(PPC_P_LOWER).toBeLessThan(0.5);
  });

  it("CI quantiles are symmetric around 0.5", () => {
    expect(CI_LOWER + CI_UPPER).toBe(1);
    expect(CI_LOWER).toBeLessThan(0.5);
  });

  it("default N samples is positive", () => {
    expect(DEFAULT_N_SAMPLES).toBeGreaterThan(0);
  });

  it("power scaling threshold is positive", () => {
    expect(POWER_SCALING_THRESHOLD).toBeGreaterThan(0);
    expect(POWER_SCALING_THRESHOLD).toBeLessThan(1);
  });
});
