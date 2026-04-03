import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/openrouter-session", () => ({
  readOpenRouterSession: vi.fn(),
}));

import { readOpenRouterSession } from "@/lib/server/openrouter-session";

const originalDeploymentEnv = process.env.DEPLOYMENT_ENV;
const originalOpenRouterApiKey = process.env.OPENROUTER_API_KEY;
const originalOpenRouterCreditsApiKey = process.env.OPENROUTER_CREDITS_API_KEY;
const originalFetch = globalThis.fetch;

describe("openrouter-access", () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.resetModules();
    globalThis.fetch = originalFetch;

    if (originalDeploymentEnv === undefined) {
      delete process.env.DEPLOYMENT_ENV;
    } else {
      process.env.DEPLOYMENT_ENV = originalDeploymentEnv;
    }

    if (originalOpenRouterApiKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalOpenRouterApiKey;
    }

    if (originalOpenRouterCreditsApiKey === undefined) {
      delete process.env.OPENROUTER_CREDITS_API_KEY;
    } else {
      process.env.OPENROUTER_CREDITS_API_KEY = originalOpenRouterCreditsApiKey;
    }
  });

  it("uses local mode outside production and ignores any stored BYOK session", async () => {
    process.env.DEPLOYMENT_ENV = "development";
    process.env.OPENROUTER_API_KEY = "local-key";
    vi.mocked(readOpenRouterSession).mockResolvedValue({
      apiKey: "user-key",
      userId: "or-user-123",
    });

    const { resolveOpenRouterAccess, toAccessStatus } = await import("./openrouter-access");

    const access = await resolveOpenRouterAccess();
    expect(access).toEqual({
      mode: "local",
      apiKey: "local-key",
    });
    expect(toAccessStatus(access)).toEqual({ mode: "local", canRun: true });
  });

  it("uses user mode in production when a BYOK session exists", async () => {
    process.env.DEPLOYMENT_ENV = "production";
    process.env.OPENROUTER_API_KEY = "anonymous-key";
    vi.mocked(readOpenRouterSession).mockResolvedValue({
      apiKey: "user-key",
      userId: "or-user-123",
    });

    const { resolveOpenRouterAccess, toAccessStatus } = await import("./openrouter-access");

    const access = await resolveOpenRouterAccess();
    expect(access).toEqual({
      mode: "user",
      apiKey: "user-key",
      userId: "or-user-123",
    });
    expect(toAccessStatus(access)).toEqual({ mode: "user", canRun: true });
  });

  it("uses anonymous mode in production when only the shared key is available", async () => {
    process.env.DEPLOYMENT_ENV = "production";
    process.env.OPENROUTER_API_KEY = "anonymous-key";
    process.env.OPENROUTER_CREDITS_API_KEY = "credits-key";
    vi.mocked(readOpenRouterSession).mockResolvedValue(null);
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ data: { total_credits: 10, total_usage: 4 } }),
    } as Response) as typeof fetch;

    const { resolveOpenRouterAccess, toAccessStatus } = await import("./openrouter-access");

    const access = await resolveOpenRouterAccess();
    expect(access).toEqual({
      mode: "anonymous",
      apiKey: "anonymous-key",
      creditStatus: "available",
    });
    expect(toAccessStatus(access)).toEqual({
      mode: "anonymous",
      canRun: true,
      creditStatus: "available",
    });
  });
});
