import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi.fn(),
  setWorkspaceAccessCookie: vi.fn((response: Response) => response),
}));

import {
  requireWorkspaceAccess,
  setWorkspaceAccessCookie,
} from "@/lib/workspace-access";
import { POST } from "./route";

describe("POST /api/workspaces/unlock", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("requires an access code", async () => {
    const response = await POST(
      new Request("http://localhost/api/workspaces/unlock", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "DEFAULT" }),
      }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({ error: "accessCode is required" });
  });

  it("sets the workspace cookie after a successful unlock", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "DEFAULT",
      setCookieCode: "test",
    });

    const response = await POST(
      new Request("http://localhost/api/workspaces/unlock", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "DEFAULT", accessCode: "test" }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true });
    expect(requireWorkspaceAccess).toHaveBeenCalledWith(
      expect.any(Request),
      "DEFAULT",
      { accessCode: "test", allowCreate: false },
    );
    expect(setWorkspaceAccessCookie).toHaveBeenCalledWith(response, "DEFAULT", "test");
  });
});
