import { describe, expect, it } from "vitest";
import { generateAnonymousWorkspaceId } from "./workspace-id";

const CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";

describe("generateAnonymousWorkspaceId", () => {
  it("only contains valid characters and has length 6", () => {
    for (let i = 0; i < 50; i++) {
      const workspaceId = generateAnonymousWorkspaceId();
      expect(workspaceId).toHaveLength(6);
      for (const ch of workspaceId) {
        expect(CHARSET).toContain(ch);
      }
    }
  });

});
