import { describe, expect, it } from "vitest";
import { unrollCausalLinks } from "./unroll";

describe("unrollCausalLinks", () => {
  it("maps lagged and contemporaneous cross-construct edges to different time slices", () => {
    const built = unrollCausalLinks(
      [
        { cause: "varying", effect: "outcome", lagged: true },
        { cause: "same_time", effect: "outcome", lagged: false },
        { cause: "stable", effect: "outcome", lagged: true },
      ],
      new Set(["varying", "same_time"]),
    );

    expect(built.ghosts).toEqual(["varying__p"]);
    expect(built.edges).toEqual([
      {
        cause: "varying",
        effect: "outcome",
        lagged: true,
        source: "varying__p",
        target: "outcome",
      },
      {
        cause: "same_time",
        effect: "outcome",
        lagged: false,
        source: "same_time",
        target: "outcome",
      },
      {
        cause: "stable",
        effect: "outcome",
        lagged: true,
        source: "stable",
        target: "outcome",
      },
    ]);
  });
});
