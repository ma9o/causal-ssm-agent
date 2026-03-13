import { describe, expect, it } from "vitest";
import { generateAnonymousUserId } from "./user-id";

const CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";

describe("generateAnonymousUserId", () => {
  it("returns a string of length 6", () => {
    const userId = generateAnonymousUserId();
    expect(userId).toHaveLength(6);
  });

  it("only contains valid characters", () => {
    for (let i = 0; i < 50; i++) {
      const userId = generateAnonymousUserId();
      for (const ch of userId) {
        expect(CHARSET).toContain(ch);
      }
    }
  });

  it("excludes ambiguous characters 0, 1, I, O", () => {
    const userIds = Array.from({ length: 100 }, () => generateAnonymousUserId()).join("");
    expect(userIds).not.toContain("0");
    expect(userIds).not.toContain("1");
    expect(userIds).not.toContain("I");
    expect(userIds).not.toContain("O");
  });

  it("generates different IDs on successive calls", () => {
    const userIds = new Set(Array.from({ length: 20 }, () => generateAnonymousUserId()));
    // With 31^6 possible IDs, collisions in 20 samples are essentially impossible.
    expect(userIds.size).toBe(20);
  });
});
