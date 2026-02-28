import { describe, expect, it } from "vitest";
import { buildHistogram } from "./histogram";

describe("buildHistogram", () => {
  it("returns empty array for empty input", () => {
    expect(buildHistogram([])).toEqual([]);
  });

  it("produces bins with correct structure", () => {
    const values = [1, 2, 3, 4, 5];
    const bins = buildHistogram(values, 5);
    expect(bins.length).toBeGreaterThan(0);
    for (const b of bins) {
      expect(b).toHaveProperty("binCenter");
      expect(b).toHaveProperty("count");
      expect(b).toHaveProperty("binStart");
      expect(b).toHaveProperty("binEnd");
      expect(typeof b.binCenter).toBe("number");
      expect(typeof b.count).toBe("number");
    }
  });

  it("total count equals input length", () => {
    const values = Array.from({ length: 100 }, (_, i) => i);
    const bins = buildHistogram(values);
    const total = bins.reduce((sum, b) => sum + b.count, 0);
    expect(total).toBe(100);
  });

  it("binStart <= binCenter <= binEnd", () => {
    const values = [1, 2, 3, 4, 5, 10, 20, 50];
    const bins = buildHistogram(values, 10);
    for (const b of bins) {
      expect(b.binStart).toBeLessThanOrEqual(b.binCenter);
      expect(b.binCenter).toBeLessThanOrEqual(b.binEnd);
    }
  });

  it("single value produces bins", () => {
    const bins = buildHistogram([42]);
    expect(bins.length).toBeGreaterThan(0);
    const total = bins.reduce((sum, b) => sum + b.count, 0);
    expect(total).toBe(1);
  });

  it("respects nBins parameter approximately", () => {
    const values = Array.from({ length: 1000 }, (_, i) => i);
    const bins5 = buildHistogram(values, 5);
    const bins50 = buildHistogram(values, 50);
    // More bins requested → more bins created (d3 may adjust slightly)
    expect(bins50.length).toBeGreaterThan(bins5.length);
  });
});
