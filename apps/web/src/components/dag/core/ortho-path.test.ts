import { describe, expect, it } from "vitest";
import { orthoPath } from "./ortho-path";

describe("orthoPath", () => {
  it("returns an empty string for fewer than two points", () => {
    expect(orthoPath([])).toBe("");
    expect(orthoPath([{ x: 1, y: 2 }])).toBe("");
  });

  it("draws a straight move+line for a two-point path", () => {
    expect(
      orthoPath([
        { x: 0, y: 0 },
        { x: 10, y: 0 },
      ]),
    ).toBe("M0,0L10,0");
  });

  it("softens an interior corner with a quadratic fillet", () => {
    const d = orthoPath([
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
    ]);
    // Starts at the first point and contains a quadratic (corner) segment.
    expect(d.startsWith("M0,0")).toBe(true);
    expect(d).toContain("Q10,0");
    expect(d.endsWith("10,10")).toBe(true);
  });

  it("caps the fillet radius at half the shorter adjacent segment", () => {
    // 4px segment → radius capped at 2, so the fillet starts 2px before the corner.
    const d = orthoPath(
      [
        { x: 0, y: 0 },
        { x: 4, y: 0 },
        { x: 4, y: 100 },
      ],
      12,
    );
    expect(d).toContain("L2.0,0");
  });
});
