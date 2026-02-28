import { describe, expect, it } from "vitest";
import { evaluatePdf } from "./distributions";

describe("evaluatePdf", () => {
  it("returns 201 points by default", () => {
    const points = evaluatePdf("normal", { mu: 0, sigma: 1 });
    expect(points).toHaveLength(201); // nPoints=200 → 0..200 inclusive
  });

  it("respects custom nPoints", () => {
    const points = evaluatePdf("normal", { mu: 0, sigma: 1 }, 50);
    expect(points).toHaveLength(51);
  });

  it("normal PDF peaks near mean", () => {
    const points = evaluatePdf("normal", { mu: 0, sigma: 1 });
    const peak = points.reduce((max, p) => (p.y > max.y ? p : max));
    expect(Math.abs(peak.x)).toBeLessThan(0.1);
  });

  it("gaussian alias works", () => {
    const points = evaluatePdf("gaussian", { mu: 0, sigma: 1 });
    expect(points.length).toBeGreaterThan(0);
    const peak = points.reduce((max, p) => (p.y > max.y ? p : max));
    expect(Math.abs(peak.x)).toBeLessThan(0.1);
  });

  it("accepts loc/scale aliases", () => {
    const points = evaluatePdf("normal", { loc: 5, scale: 2 });
    const peak = points.reduce((max, p) => (p.y > max.y ? p : max));
    expect(Math.abs(peak.x - 5)).toBeLessThan(0.2);
  });

  it("half-normal has zero density for negative x", () => {
    const points = evaluatePdf("halfnormal", { sigma: 1 });
    expect(points.every((p) => p.x >= 0)).toBe(true);
  });

  it("half_normal alias works", () => {
    const points = evaluatePdf("half_normal", { sigma: 1 });
    expect(points.every((p) => p.x >= 0)).toBe(true);
  });

  it("gamma has only positive x values", () => {
    const points = evaluatePdf("gamma", { alpha: 2, beta: 1 });
    expect(points.every((p) => p.x >= 0)).toBe(true);
  });

  it("beta is bounded in (0, 1)", () => {
    const points = evaluatePdf("beta", { alpha: 2, beta: 5 });
    expect(points.every((p) => p.x >= 0 && p.x <= 1)).toBe(true);
  });

  it("uniform PDF is constant within bounds", () => {
    const points = evaluatePdf("uniform", { low: 0, high: 1 });
    const interior = points.filter((p) => p.x > 0.05 && p.x < 0.95);
    const yValues = interior.map((p) => p.y);
    const avg = yValues.reduce((s, v) => s + v, 0) / yValues.length;
    // Should be close to 1/(1-0) = 1
    expect(Math.abs(avg - 1)).toBeLessThan(0.01);
  });

  it("unknown distribution returns zero densities", () => {
    const points = evaluatePdf("unknown_dist", {});
    expect(points.every((p) => p.y === 0)).toBe(true);
  });

  it("all points have numeric x and y", () => {
    for (const dist of ["normal", "halfnormal", "gamma", "beta", "uniform"]) {
      const points = evaluatePdf(dist, { mu: 0, sigma: 1, alpha: 2, beta: 1, low: 0, high: 1 });
      for (const p of points) {
        expect(typeof p.x).toBe("number");
        expect(typeof p.y).toBe("number");
        expect(Number.isFinite(p.x)).toBe(true);
        expect(Number.isFinite(p.y)).toBe(true);
      }
    }
  });
});
