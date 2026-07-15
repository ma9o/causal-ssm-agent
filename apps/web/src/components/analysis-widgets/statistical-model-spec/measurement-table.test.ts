import { describe, expect, it } from "vitest";
import { binPriorSamples } from "./measurement-table";

describe("binPriorSamples", () => {
  it("does not clamp continuous predictive tail mass into boundary bins", () => {
    const bins = [
      { binCenter: 0, count: 2, binStart: 0, binEnd: 0 },
      { binCenter: 1, count: 2, binStart: 1, binEnd: 1 },
    ];

    const prior = binPriorSamples([-10, 0, 1, 10], bins, 4, false);

    expect(prior).toEqual([
      { binCenter: 0, prior: 1 },
      { binCenter: 1, prior: 1 },
    ]);
  });

  it("keeps discrete predictive mass on exact declared bins", () => {
    const bins = [
      { binCenter: 0, count: 2, binStart: 0, binEnd: 0 },
      { binCenter: 1, count: 2, binStart: 1, binEnd: 1 },
      { binCenter: 2, count: 0, binStart: 2, binEnd: 2 },
    ];

    const prior = binPriorSamples([0, 1, 2, 2], bins, 4, true);

    expect(prior.map((entry) => entry.prior)).toEqual([1, 1, 2]);
  });
});
