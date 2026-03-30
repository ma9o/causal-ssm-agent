import { describe, expect, it } from "vitest";
import { evaluatePdf } from "./distributions";

describe("evaluatePdf", () => {
  it("returns nPoints+1 data points", () => {
    const points = evaluatePdf("Normal", { mu: 0, sigma: 1 }, 100);
    expect(points).toHaveLength(101);
  });

  describe("Normal distribution", () => {
    it("peaks near mu", () => {
      const points = evaluatePdf("Normal", { mu: 5, sigma: 1 });
      const peak = points.reduce((a, b) => (a.y > b.y ? a : b));
      expect(peak.x).toBeCloseTo(5, 0);
    });

    it("has peak height consistent with 1/(sigma*sqrt(2pi))", () => {
      const sigma = 2;
      const points = evaluatePdf("Normal", { mu: 0, sigma });
      const peak = points.reduce((a, b) => (a.y > b.y ? a : b));
      const expectedPeak = 1 / (sigma * Math.sqrt(2 * Math.PI));
      expect(peak.y).toBeCloseTo(expectedPeak, 2);
    });

    it("uses canonical mu/sigma params", () => {
      const points = evaluatePdf("Normal", { mu: 3, sigma: 0.5 });
      const peak = points.reduce((a, b) => (a.y > b.y ? a : b));
      expect(peak.x).toBeCloseTo(3, 0);
    });
  });

  describe("HalfNormal distribution", () => {
    it("has zero density for negative x", () => {
      const points = evaluatePdf("HalfNormal", { sigma: 1 });
      const negativePoints = points.filter((p) => p.x < 0);
      for (const p of negativePoints) {
        expect(p.y).toBe(0);
      }
    });

    it("starts at x=0", () => {
      const points = evaluatePdf("HalfNormal", { sigma: 2 });
      expect(points[0].x).toBe(0);
    });

  });

  describe("Beta distribution", () => {
    it("stays within [0, 1]", () => {
      const points = evaluatePdf("Beta", { alpha: 2, beta: 5 });
      for (const p of points) {
        expect(p.x).toBeGreaterThanOrEqual(0);
        expect(p.x).toBeLessThanOrEqual(1);
      }
    });

    it("peaks at (alpha-1)/(alpha+beta-2) for symmetric Beta(2,2)", () => {
      const points = evaluatePdf("Beta", { alpha: 2, beta: 2 });
      const peak = points.reduce((a, b) => (a.y > b.y ? a : b));
      expect(peak.x).toBeCloseTo(0.5, 1);
    });
  });

  describe("Gamma distribution", () => {
    it("peaks near (concentration-1)/rate", () => {
      const points = evaluatePdf("Gamma", { concentration: 3, rate: 2 });
      const peak = points.reduce((a, b) => (a.y > b.y ? a : b));
      // mode = (3-1)/2 = 1.0
      expect(peak.x).toBeCloseTo(1.0, 0);
    });

    it("starts at x=0", () => {
      const points = evaluatePdf("Gamma", { concentration: 2, rate: 1 });
      expect(points[0].x).toBe(0);
    });
  });

  describe("LogNormal distribution", () => {
    it("peaks near exp(mu - sigma^2) for mu=0, sigma=0.5", () => {
      const points = evaluatePdf("LogNormal", { mu: 0, sigma: 0.5 });
      const peak = points.reduce((a, b) => (a.y > b.y ? a : b));
      // mode = exp(0 - 0.25) ≈ 0.779
      expect(peak.x).toBeCloseTo(Math.exp(-0.25), 0);
    });

    it("starts at x=0 and has zero density there", () => {
      const points = evaluatePdf("LogNormal", { mu: 0, sigma: 0.5 });
      expect(points[0].x).toBe(0);
      expect(points[0].y).toBe(0);
    });
  });

  describe("Exponential distribution", () => {
    it("peaks at x=0 with density equal to rate", () => {
      const rate = 2;
      const points = evaluatePdf("Exponential", { rate });
      expect(points[0].x).toBe(0);
      expect(points[0].y).toBeCloseTo(rate, 5);
    });

    it("decays monotonically", () => {
      const points = evaluatePdf("Exponential", { rate: 2 });
      for (let i = 1; i < points.length; i++) {
        expect(points[i].y).toBeLessThanOrEqual(points[i - 1].y + 1e-10);
      }
    });
  });

  describe("TruncatedNormal distribution", () => {
    it("respects the provided lower and upper bounds", () => {
      const points = evaluatePdf("TruncatedNormal", { mu: 0, sigma: 1, lower: -1, upper: 1 });
      expect(points[0].x).toBe(-1);
      expect(points[points.length - 1].x).toBe(1);
    });
  });

  describe("Uniform distribution", () => {
    it("is flat between low and high", () => {
      const points = evaluatePdf("Uniform", { lower: 2, upper: 5 }, 300);
      const interiorPoints = points.filter((p) => p.x > 2.1 && p.x < 4.9);
      const expectedY = 1 / (5 - 2);
      for (const p of interiorPoints) {
        expect(p.y).toBeCloseTo(expectedY, 3);
      }
    });

    it("is zero outside bounds", () => {
      const points = evaluatePdf("Uniform", { lower: 2, upper: 5 }, 300);
      const outsidePoints = points.filter((p) => p.x < 2 || p.x > 5);
      for (const p of outsidePoints) {
        expect(p.y).toBe(0);
      }
    });

    it("handles degenerate case where low equals high without Infinity", () => {
      const points = evaluatePdf("Uniform", { lower: 3, upper: 3 });
      for (const p of points) {
        expect(Number.isFinite(p.y)).toBe(true);
        expect(p.y).toBe(0);
      }
    });
  });

  it("uses default nPoints=200", () => {
    const points = evaluatePdf("Normal", { mu: 0, sigma: 1 });
    expect(points).toHaveLength(201);
  });

  it("handles unknown distribution gracefully", () => {
    const points = evaluatePdf("StudentT", { mu: 0, sigma: 1 });
    for (const p of points) {
      expect(p.y).toBe(0);
    }
  });

  it("rejects non-canonical distribution spellings by treating them as unknown", () => {
    const points = evaluatePdf("gaussian", { mu: 0, sigma: 1 });
    for (const p of points) {
      expect(p.y).toBe(0);
    }
  });
});
