import { describe, expect, it } from "vitest";
import { generateAnonymousWorkspaceId } from "./workspace-id";

const CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";

describe("generateAnonymousWorkspaceId", () => {
  it("returns a string of length 6", () => {
    const workspaceId = generateAnonymousWorkspaceId();
    expect(workspaceId).toHaveLength(6);
  });

  it("only contains valid characters", () => {
    for (let i = 0; i < 50; i++) {
      const workspaceId = generateAnonymousWorkspaceId();
      for (const ch of workspaceId) {
        expect(CHARSET).toContain(ch);
      }
    }
  });

  it("excludes ambiguous characters 0, 1, I, O", () => {
    const workspaceIds = Array.from({ length: 100 }, () => generateAnonymousWorkspaceId()).join("");
    expect(workspaceIds).not.toContain("0");
    expect(workspaceIds).not.toContain("1");
    expect(workspaceIds).not.toContain("I");
    expect(workspaceIds).not.toContain("O");
  });
});
