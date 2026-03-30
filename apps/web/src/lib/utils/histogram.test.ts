import { describe, expect, it } from "vitest";
import { buildHistogram, quantile } from "./histogram";

describe("buildHistogram", () => {
  it("returns empty array for empty input", () => {
    expect(buildHistogram([])).toEqual([]);
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

  it("handles negative numbers", () => {
    const values = [-10, -5, 0, 5, 10];
    const bins = buildHistogram(values, 5);
    const total = bins.reduce((sum, b) => sum + b.count, 0);
    expect(total).toBe(5);
    expect(bins[0].binStart).toBeLessThan(0);
  });

  it("handles identical values", () => {
    const values = [7, 7, 7, 7, 7];
    const bins = buildHistogram(values, 5);
    const total = bins.reduce((sum, b) => sum + b.count, 0);
    expect(total).toBe(5);
  });

  it("handles nBins greater than array length", () => {
    const values = [1, 2, 3];
    const bins = buildHistogram(values, 100);
    const total = bins.reduce((sum, b) => sum + b.count, 0);
    expect(total).toBe(3);
  });
});

describe("quantile", () => {
  it("returns exact value at q=0", () => {
    expect(quantile([1, 2, 3, 4, 5], 0)).toBe(1);
  });

  it("returns exact value at q=1", () => {
    expect(quantile([1, 2, 3, 4, 5], 1)).toBe(5);
  });

  it("returns median at q=0.5 for odd-length array", () => {
    expect(quantile([1, 2, 3, 4, 5], 0.5)).toBe(3);
  });

  it("interpolates for even-length array at q=0.5", () => {
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5);
  });

  it("returns single value for single-element array", () => {
    expect(quantile([42], 0)).toBe(42);
    expect(quantile([42], 0.5)).toBe(42);
    expect(quantile([42], 1)).toBe(42);
  });

  it("handles q=0.025 for credible intervals", () => {
    const sorted = Array.from({ length: 1000 }, (_, i) => i / 999);
    const lo = quantile(sorted, 0.025);
    expect(lo).toBeCloseTo(0.025, 2);
  });

  it("handles q=0.975 for credible intervals", () => {
    const sorted = Array.from({ length: 1000 }, (_, i) => i / 999);
    const hi = quantile(sorted, 0.975);
    expect(hi).toBeCloseTo(0.975, 2);
  });

  it("returns exact value when q lands on an index", () => {
    // q=0.25 on [0, 10, 20, 30, 40] → index 1 → value 10
    expect(quantile([0, 10, 20, 30, 40], 0.25)).toBe(10);
  });

  it("returns NaN for empty array", () => {
    expect(quantile([], 0.5)).toBeNaN();
  });
});
