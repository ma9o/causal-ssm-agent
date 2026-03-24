import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/openrouter-session", () => ({
  hasOpenRouterSessionSecret: vi.fn(),
  createOpenRouterSession: vi.fn(),
  writeOpenRouterSession: vi.fn(),
}));

import {
  createOpenRouterSession,
  hasOpenRouterSessionSecret,
  writeOpenRouterSession,
} from "@/lib/server/openrouter-session";
import { POST } from "./route";

const originalFetch = globalThis.fetch;

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
  } as Response;
}

describe("POST /api/auth/exchange", () => {
  afterEach(() => {
    vi.clearAllMocks();
    globalThis.fetch = originalFetch;
  });

  it("requires a code verifier", async () => {
    const response = await POST(
      new Request("http://localhost/api/auth/exchange", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: "auth-code" }),
      }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({ error: "Missing PKCE code verifier" });
  });

  it("sets the encrypted session cookie after a successful exchange", async () => {
    vi.mocked(hasOpenRouterSessionSecret).mockReturnValue(true);
    vi.mocked(createOpenRouterSession).mockReturnValue({
      apiKey: "user-key",
    });
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse({ key: "user-key", user_id: "or-user-123" }),
    ) as typeof fetch;

    const response = await POST(
      new Request("http://localhost/api/auth/exchange", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: "auth-code", code_verifier: "pkce-verifier" }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true });
    expect(globalThis.fetch).toHaveBeenCalledWith(
      "https://openrouter.ai/api/v1/auth/keys",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          code: "auth-code",
          code_verifier: "pkce-verifier",
          code_challenge_method: "S256",
        }),
      }),
    );
    expect(createOpenRouterSession).toHaveBeenCalledWith("user-key");
    expect(writeOpenRouterSession).toHaveBeenCalledWith({
      apiKey: "user-key",
    });
  });
});
