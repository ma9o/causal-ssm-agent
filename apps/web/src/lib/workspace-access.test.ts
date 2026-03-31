import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/workspace-session", () => ({
  authorizeWorkspaceInSession: vi.fn(),
  hasWorkspaceSessionAccess: vi.fn(),
}));

vi.mock("@/lib/storage", () => ({
  prefixExists: vi.fn(),
}));

import { authorizeWorkspaceInSession, hasWorkspaceSessionAccess } from "@/lib/server/workspace-session";
import { prefixExists } from "@/lib/storage";
import { requireWorkspaceAccess } from "./workspace-access";

describe("requireWorkspaceAccess", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("rejects invalid workspace ids", async () => {
    const result = await requireWorkspaceAccess(new Request("http://localhost"), "bad/workspace");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.response.status).toBe(400);
    }
  });

  it("allows shared fixture workspaces without a session", async () => {
    const result = await requireWorkspaceAccess(new Request("http://localhost"), "DEFAULT");

    expect(result).toEqual({
      ok: true,
      workspaceId: "DEFAULT",
    });
    expect(hasWorkspaceSessionAccess).not.toHaveBeenCalled();
  });

  it("allows workspaces already present in the browser session", async () => {
    vi.mocked(hasWorkspaceSessionAccess).mockResolvedValue(true);

    const result = await requireWorkspaceAccess(new Request("http://localhost"), "USER123");

    expect(result).toEqual({
      ok: true,
      workspaceId: "USER123",
    });
    expect(prefixExists).not.toHaveBeenCalled();
  });

  it("claims a fresh workspace for the current browser session", async () => {
    vi.mocked(hasWorkspaceSessionAccess).mockResolvedValue(false);
    vi.mocked(prefixExists).mockResolvedValue(false);

    const result = await requireWorkspaceAccess(new Request("http://localhost"), "NEWSPACE", {
      allowCreate: true,
    });

    expect(result).toEqual({
      ok: true,
      workspaceId: "NEWSPACE",
    });
    expect(prefixExists).toHaveBeenCalledWith("NEWSPACE/");
    expect(authorizeWorkspaceInSession).toHaveBeenCalledWith("NEWSPACE");
  });

  it("refuses to claim an existing workspace for a new browser session", async () => {
    vi.mocked(hasWorkspaceSessionAccess).mockResolvedValue(false);
    vi.mocked(prefixExists).mockResolvedValue(true);

    const result = await requireWorkspaceAccess(new Request("http://localhost"), "USER123", {
      allowCreate: true,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.response.status).toBe(403);
    }
    expect(authorizeWorkspaceInSession).not.toHaveBeenCalled();
  });
});
