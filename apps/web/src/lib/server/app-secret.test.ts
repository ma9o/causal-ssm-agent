import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const originalCwd = process.cwd();
const originalNodeEnv = process.env.NODE_ENV;
const originalAppSecret = process.env.APP_SECRET;

describe("app secret", () => {
  let tempDir: string | null = null;

  afterEach(() => {
    vi.resetModules();
    process.chdir(originalCwd);

    if (originalNodeEnv === undefined) {
      delete process.env.NODE_ENV;
    } else {
      process.env.NODE_ENV = originalNodeEnv;
    }

    if (originalAppSecret === undefined) {
      delete process.env.APP_SECRET;
    } else {
      process.env.APP_SECRET = originalAppSecret;
    }

    if (tempDir) {
      rmSync(tempDir, { recursive: true, force: true });
      tempDir = null;
    }
  });

  it("derives stable scoped secrets from APP_SECRET", async () => {
    tempDir = mkdtempSync(join(tmpdir(), "app-secret-env-"));
    mkdirSync(join(tempDir, "apps"));
    mkdirSync(join(tempDir, "packages"));
    process.chdir(tempDir);
    process.env.NODE_ENV = "test";
    process.env.APP_SECRET = "0123456789abcdef0123456789abcdef";

    const { deriveAppSecret } = await import("./app-secret");

    expect(deriveAppSecret("openrouter-session")).toHaveLength(64);
    expect(deriveAppSecret("byok-secret-store")).toHaveLength(64);
    expect(deriveAppSecret("openrouter-session")).not.toBe(
      deriveAppSecret("byok-secret-store"),
    );
  });

  it("fails fast when APP_SECRET is missing", async () => {
    tempDir = mkdtempSync(join(tmpdir(), "app-secret-missing-"));
    mkdirSync(join(tempDir, "apps"));
    mkdirSync(join(tempDir, "packages"));
    process.chdir(tempDir);
    process.env.NODE_ENV = "test";
    delete process.env.APP_SECRET;

    const { getAppSecret } = await import("./app-secret");

    expect(() => getAppSecret()).toThrow("APP_SECRET is not configured");
  });
});
