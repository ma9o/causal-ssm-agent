import { describe, expect, it } from "vitest";
import { CHAIN_COLORS, diagnosisBadgeVariant, diagnosisColor, diagnosisLabel } from "./charts";

describe("CHAIN_COLORS", () => {
  it("has at least 4 colors for multi-chain plots", () => {
    expect(CHAIN_COLORS.length).toBeGreaterThanOrEqual(4);
  });

  it("contains CSS variable references", () => {
    for (const color of CHAIN_COLORS) {
      expect(color).toMatch(/^var\(--/);
    }
  });
});

describe("diagnosisLabel", () => {
  it("maps all three diagnosis types", () => {
    expect(diagnosisLabel.well_identified).toBeDefined();
    expect(diagnosisLabel.prior_dominated).toBeDefined();
    expect(diagnosisLabel.prior_data_conflict).toBeDefined();
  });

  it("labels are human-readable strings", () => {
    for (const label of Object.values(diagnosisLabel)) {
      expect(typeof label).toBe("string");
      expect(label.length).toBeGreaterThan(0);
    }
  });
});

describe("diagnosisColor", () => {
  it("maps all three diagnosis types to CSS variables", () => {
    expect(diagnosisColor.well_identified).toMatch(/^var\(--/);
    expect(diagnosisColor.prior_dominated).toMatch(/^var\(--/);
    expect(diagnosisColor.prior_data_conflict).toMatch(/^var\(--/);
  });
});

describe("diagnosisBadgeVariant", () => {
  it("maps diagnoses to valid badge variants", () => {
    const validVariants = new Set(["success", "warning", "destructive"]);
    for (const variant of Object.values(diagnosisBadgeVariant)) {
      expect(validVariants.has(variant)).toBe(true);
    }
  });

  it("well_identified is success", () => {
    expect(diagnosisBadgeVariant.well_identified).toBe("success");
  });

  it("prior_dominated is warning", () => {
    expect(diagnosisBadgeVariant.prior_dominated).toBe("warning");
  });

  it("prior_data_conflict is destructive", () => {
    expect(diagnosisBadgeVariant.prior_data_conflict).toBe("destructive");
  });

  it("all three maps have consistent keys", () => {
    const labelKeys = Object.keys(diagnosisLabel).sort();
    const colorKeys = Object.keys(diagnosisColor).sort();
    const badgeKeys = Object.keys(diagnosisBadgeVariant).sort();
    expect(labelKeys).toEqual(colorKeys);
    expect(colorKeys).toEqual(badgeKeys);
  });
});
