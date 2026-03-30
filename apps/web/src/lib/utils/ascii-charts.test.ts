import { describe, expect, it } from "vitest";
import { asciiDensity, asciiHistogram, asciiMultiLine, asciiScatter } from "./ascii-charts";

describe("asciiHistogram", () => {
  it("returns no data for empty array", () => {
    expect(asciiHistogram([])).toBe("(no data)");
  });

  it("returns constant message when all values equal", () => {
    expect(asciiHistogram([5, 5, 5])).toBe("All values = 5.000");
  });

  it("renders a histogram with data", () => {
    const values = Array.from({ length: 100 }, (_, i) => i);
    const result = asciiHistogram(values);
    expect(result).toContain("│");
    expect(result.split("\n").length).toBeGreaterThan(1);
  });

  it("includes label when provided", () => {
    const result = asciiHistogram([1, 2, 3, 4, 5], { label: "My Histogram" });
    expect(result).toContain("My Histogram");
  });

  it("respects nBins option", () => {
    const values = Array.from({ length: 100 }, (_, i) => i);
    const result = asciiHistogram(values, { nBins: 5 });
    const barLines = result.split("\n").filter((l) => l.includes("│"));
    expect(barLines).toHaveLength(5);
  });

  it("includes summary stats", () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = asciiHistogram(values);
    expect(result).toContain("n=10");
    expect(result).toContain("mean=");
    expect(result).toContain("sd=");
    expect(result).toContain("median=");
    expect(result).toContain("range=");
  });
});

describe("asciiDensity", () => {
  it("returns no data for empty array", () => {
    expect(asciiDensity([], [])).toBe("(no data)");
  });

  it("renders density with data", () => {
    const x = Array.from({ length: 50 }, (_, i) => i * 0.1);
    const y = x.map((v) => Math.exp(-v * v));
    const result = asciiDensity(x, y);
    expect(result).toContain("x:");
  });

  it("includes label when provided", () => {
    const x = [0, 1, 2];
    const y = [0.5, 1.0, 0.5];
    const result = asciiDensity(x, y, { label: "Density" });
    expect(result).toContain("Density");
  });

  it("includes density-weighted stats", () => {
    const x = [0, 1, 2, 3, 4];
    const y = [0.1, 0.2, 0.5, 0.2, 0.1];
    const result = asciiDensity(x, y);
    expect(result).toContain("mean=");
    expect(result).toContain("sd=");
    expect(result).toContain("mode=");
  });
});

describe("asciiScatter", () => {
  it("returns no data for empty array", () => {
    expect(asciiScatter([])).toBe("(no data)");
  });

  it("renders scatter plot with points", () => {
    const points = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
    ];
    const result = asciiScatter(points);
    expect(result).toContain("•");
    expect(result).toContain("│");
  });

  it("handles single point", () => {
    const result = asciiScatter([{ x: 5, y: 5 }]);
    expect(result).toContain("•");
  });

  it("includes label when provided", () => {
    const result = asciiScatter([{ x: 0, y: 0 }], { label: "Plot" });
    expect(result).toContain("Plot");
  });

  it("includes summary stats for both axes", () => {
    const points = [
      { x: 1, y: 10 },
      { x: 2, y: 20 },
      { x: 3, y: 30 },
    ];
    const result = asciiScatter(points);
    expect(result).toContain("n=3");
    expect(result).toContain("x: mean=");
    expect(result).toContain("y: mean=");
  });
});

describe("asciiMultiLine", () => {
  it("returns no data for empty series", () => {
    expect(asciiMultiLine([])).toBe("(no data)");
  });

  it("returns no data when all series are empty", () => {
    expect(asciiMultiLine([[], []])).toBe("(no data)");
  });

  it("renders with a single series", () => {
    const result = asciiMultiLine([[1, 2, 3, 4, 5]]);
    expect(result).toContain("│");
    expect(result).toContain("series 1");
    expect(result).toContain("mean=");
  });

  it("renders with multiple series", () => {
    const result = asciiMultiLine([
      [1, 2, 3, 4, 5],
      [5, 4, 3, 2, 1],
    ]);
    expect(result).toContain("│");
    expect(result).toContain("series 1");
    expect(result).toContain("series 2");
  });

  it("includes label when provided", () => {
    const result = asciiMultiLine([[1, 2, 3]], { label: "Traces" });
    expect(result).toContain("Traces");
  });

  it("handles width=1 without NaN from resample", () => {
    const long = Array.from({ length: 100 }, (_, i) => Math.sin(i));
    const result = asciiMultiLine([long], { width: 1 });
    expect(result).not.toContain("NaN");
  });

  it("includes per-series summary stats", () => {
    const result = asciiMultiLine([
      [1, 2, 3, 4, 5],
      [10, 20, 30, 40, 50],
    ]);
    expect(result).toContain("series 1");
    expect(result).toContain("series 2");
    expect(result).toContain("mean=");
    expect(result).toContain("sd=");
  });
});

describe("asciiScatter edge cases", () => {
  it("handles all-same x values without division by zero", () => {
    const points = [
      { x: 5, y: 1 },
      { x: 5, y: 2 },
      { x: 5, y: 3 },
    ];
    const result = asciiScatter(points);
    expect(result).not.toContain("NaN");
    expect(result).toContain("•");
  });

  it("handles all-same y values without division by zero", () => {
    const points = [
      { x: 1, y: 5 },
      { x: 2, y: 5 },
      { x: 3, y: 5 },
    ];
    const result = asciiScatter(points);
    expect(result).not.toContain("NaN");
    expect(result).toContain("•");
  });

  it("handles height=1 without NaN or errors", () => {
    const points = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ];
    const result = asciiScatter(points, { height: 1 });
    expect(result).not.toContain("NaN");
    expect(result).not.toContain("Infinity");
    expect(result).toContain("•");
  });
});
