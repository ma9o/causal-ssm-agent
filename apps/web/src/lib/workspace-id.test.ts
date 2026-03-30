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

});
