import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/openrouter-access", () => ({
  getOpenRouterStatus: vi.fn(),
}));

import { getOpenRouterStatus } from "@/lib/server/openrouter-access";
import { GET } from "./route";

describe("GET /api/auth/status", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("returns the server-derived access status", async () => {
    vi.mocked(getOpenRouterStatus).mockResolvedValue({
      mode: "trial",
      canRun: true,
      creditStatus: "unknown",
    });

    const response = await GET();

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      mode: "trial",
      canRun: true,
      creditStatus: "unknown",
    });
    expect(getOpenRouterStatus).toHaveBeenCalledWith();
  });
});
