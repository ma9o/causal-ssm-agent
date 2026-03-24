import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/openrouter-session", () => ({
  clearOpenRouterSession: vi.fn(),
}));

import { clearOpenRouterSession } from "@/lib/server/openrouter-session";
import { POST } from "./route";

describe("POST /api/auth/logout", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("clears the session cookie", async () => {
    const response = await POST();

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true });
    expect(clearOpenRouterSession).toHaveBeenCalled();
  });
});
