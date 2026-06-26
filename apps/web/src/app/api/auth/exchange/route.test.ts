import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/openrouter-session", () => ({
  hasOpenRouterSessionSecret: vi.fn(),
  createOpenRouterSession: vi.fn(),
  writeOpenRouterSession: vi.fn(),
}));

vi.mock("@/lib/server/workspace-ownership", () => ({
  authorizeWorkspacesForOpenRouterUser: vi.fn(),
}));

vi.mock("@/lib/server/workspace-session", () => ({
  clearAuthorizedWorkspaceIds: vi.fn(),
  readAuthorizedWorkspaceIds: vi.fn(),
}));

import {
  createOpenRouterSession,
  hasOpenRouterSessionSecret,
  writeOpenRouterSession,
} from "@/lib/server/openrouter-session";
import { authorizeWorkspacesForOpenRouterUser } from "@/lib/server/workspace-ownership";
import {
  clearAuthorizedWorkspaceIds,
  readAuthorizedWorkspaceIds,
} from "@/lib/server/workspace-session";
import { POST } from "./route";

const originalFetch = globalThis.fetch;
const originalDeploymentEnv = process.env.DEPLOYMENT_ENV;

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
    if (originalDeploymentEnv === undefined) {
      delete process.env.DEPLOYMENT_ENV;
    } else {
      process.env.DEPLOYMENT_ENV = originalDeploymentEnv;
    }
  });

  it("requires a code verifier", async () => {
    process.env.DEPLOYMENT_ENV = "production";

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
    process.env.DEPLOYMENT_ENV = "production";
    vi.mocked(hasOpenRouterSessionSecret).mockReturnValue(true);
    vi.mocked(readAuthorizedWorkspaceIds).mockResolvedValue(["WS1", "WS2"]);
    vi.mocked(createOpenRouterSession).mockReturnValue({
      apiKey: "user-key",
      userId: "or-user-123",
    });
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(jsonResponse({ key: "user-key", user_id: "or-user-123" })) as typeof fetch;

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
    expect(createOpenRouterSession).toHaveBeenCalledWith("user-key", "or-user-123");
    expect(authorizeWorkspacesForOpenRouterUser).toHaveBeenCalledWith("or-user-123", [
      "WS1",
      "WS2",
    ]);
    expect(writeOpenRouterSession).toHaveBeenCalledWith({
      apiKey: "user-key",
      userId: "or-user-123",
    });
    expect(clearAuthorizedWorkspaceIds).toHaveBeenCalledWith();
  });

  it("disables OpenRouter exchange outside production", async () => {
    process.env.DEPLOYMENT_ENV = "development";

    const response = await POST(
      new Request("http://localhost/api/auth/exchange", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: "auth-code", code_verifier: "pkce-verifier" }),
      }),
    );

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({
      error: "OpenRouter sign-in is only available in production.",
    });
    expect(writeOpenRouterSession).not.toHaveBeenCalled();
  });
});
