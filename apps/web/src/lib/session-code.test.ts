import { describe, expect, it } from "vitest";
import { generateSessionCode } from "./session-code";

const CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";

describe("generateSessionCode", () => {
  it("returns a string of length 6", () => {
    const code = generateSessionCode();
    expect(code).toHaveLength(6);
  });

  it("only contains valid characters", () => {
    for (let i = 0; i < 50; i++) {
      const code = generateSessionCode();
      for (const ch of code) {
        expect(CHARSET).toContain(ch);
      }
    }
  });

  it("excludes ambiguous characters 0, 1, I, O", () => {
    const codes = Array.from({ length: 100 }, () => generateSessionCode()).join("");
    expect(codes).not.toContain("0");
    expect(codes).not.toContain("1");
    expect(codes).not.toContain("I");
    expect(codes).not.toContain("O");
  });

  it("generates different codes on successive calls", () => {
    const codes = new Set(Array.from({ length: 20 }, () => generateSessionCode()));
    // With 31^6 possible codes, collisions in 20 samples are essentially impossible
    expect(codes.size).toBe(20);
  });
});
