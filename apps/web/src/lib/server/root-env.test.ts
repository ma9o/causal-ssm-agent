import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const originalCwd = process.cwd();
const originalOpenRouterApiKey = process.env.OPENROUTER_API_KEY;

describe("root env loader", () => {
  let tempDir: string | null = null;

  afterEach(() => {
    vi.resetModules();
    process.chdir(originalCwd);

    if (originalOpenRouterApiKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalOpenRouterApiKey;
    }

    if (tempDir) {
      rmSync(tempDir, { recursive: true, force: true });
      tempDir = null;
    }
  });

  it("loads OPENROUTER_API_KEY from the repo root when cwd is apps/web", async () => {
    tempDir = mkdtempSync(join(tmpdir(), "root-env-"));
    mkdirSync(join(tempDir, "apps", "web"), { recursive: true });
    mkdirSync(join(tempDir, "packages"));
    writeFileSync(
      join(tempDir, ".env"),
      "OPENROUTER_API_KEY=repo-root-key\n",
      "utf8",
    );

    process.chdir(join(tempDir, "apps", "web"));
    delete process.env.OPENROUTER_API_KEY;

    await import("./root-env");

    expect(process.env.OPENROUTER_API_KEY).toBe("repo-root-key");
  });
});
