import { afterEach, describe, expect, it, vi } from "vitest";
vi.mock("node:fs", () => ({
  readFileSync: vi.fn(),
}));

import { readFileSync } from "node:fs";
import { getDefaultApiKey, resolveApiKey } from "./resolve-api-key";

describe("resolve-api-key", () => {
  const originalKey = process.env.OPENROUTER_API_KEY;

  afterEach(() => {
    if (originalKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalKey;
    }
    vi.restoreAllMocks();
  });

  it("prefers the runtime environment key", () => {
    process.env.OPENROUTER_API_KEY = "env-key";

    expect(getDefaultApiKey()).toBe("env-key");
  });

  it("falls back to the monorepo root .env file", () => {
    delete process.env.OPENROUTER_API_KEY;
    vi.mocked(readFileSync).mockReturnValue("OPENROUTER_API_KEY=file-key\n");

    expect(getDefaultApiKey()).toBe("file-key");
  });

  it("prefers a request-scoped key over the default key", () => {
    process.env.OPENROUTER_API_KEY = "env-key";
    const request = new Request("http://example.test", {
      headers: { "x-openrouter-key": "user-key" },
    });

    expect(resolveApiKey(request)).toEqual({ key: "user-key" });
  });

  it("returns 402 when no key is available", () => {
    delete process.env.OPENROUTER_API_KEY;
    vi.mocked(readFileSync).mockImplementation(() => {
      throw new Error("missing");
    });

    const request = new Request("http://example.test");
    expect(resolveApiKey(request)).toEqual({
      error: "No API key available",
      status: 402,
    });
  });
});
