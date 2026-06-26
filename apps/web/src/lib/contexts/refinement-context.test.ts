import { describe, expect, it } from "vitest";
import { refinementNeedsActivation, refinementRequiresConfirmation } from "./refinement-context";

describe("refinement helpers", () => {
  it("keeps terminal stage activation separate from downstream invalidation confirmation", () => {
    expect(refinementNeedsActivation("stage-6", null)).toBe(true);
    expect(refinementRequiresConfirmation("stage-6", null)).toBe(false);
    expect(refinementNeedsActivation("stage-6", "stage-6")).toBe(false);

    expect(refinementRequiresConfirmation("stage-1a", null)).toBe(true);
    expect(refinementRequiresConfirmation("stage-1a", "stage-6")).toBe(true);
  });
});
