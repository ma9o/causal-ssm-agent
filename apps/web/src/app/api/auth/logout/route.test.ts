import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/openrouter-session", () => ({
  clearOpenRouterSession: vi.fn(),
}));

vi.mock("@/lib/server/workspace-session", () => ({
  clearAuthorizedWorkspaceIds: vi.fn(),
}));

import { clearOpenRouterSession } from "@/lib/server/openrouter-session";
import { clearAuthorizedWorkspaceIds } from "@/lib/server/workspace-session";
import { POST } from "./route";

describe("POST /api/auth/logout", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("clears both the OpenRouter session and anonymous workspace session", async () => {
    const response = await POST();

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true });
    expect(clearAuthorizedWorkspaceIds).toHaveBeenCalledWith();
    expect(clearOpenRouterSession).toHaveBeenCalledWith();
  });
});
