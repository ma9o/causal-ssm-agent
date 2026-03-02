import { describe, expect, it } from "vitest";
import { cn } from "./cn";

describe("cn", () => {
  it("merges simple class names", () => {
    expect(cn("foo", "bar")).toBe("foo bar");
  });

  it("handles conditional classes", () => {
    expect(cn("base", false && "hidden", "visible")).toBe("base visible");
  });

  it("merges conflicting Tailwind classes", () => {
    // twMerge should resolve conflicts: later class wins
    expect(cn("p-4", "p-2")).toBe("p-2");
  });

  it("handles arrays", () => {
    expect(cn(["foo", "bar"])).toBe("foo bar");
  });

  it("handles undefined and null", () => {
    expect(cn("a", undefined, null, "b")).toBe("a b");
  });

  it("handles empty input", () => {
    expect(cn()).toBe("");
  });

  it("handles object syntax for conditional classes", () => {
    expect(cn({ "text-red-500": true, hidden: false })).toBe("text-red-500");
  });

  it("handles mixed strings and objects", () => {
    expect(cn("base", { "p-4": true, "m-2": false }, "end")).toBe("base p-4 end");
  });
});
