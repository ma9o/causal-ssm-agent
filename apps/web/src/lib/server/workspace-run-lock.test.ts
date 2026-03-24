import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { claimWorkspaceRunSlot, releaseWorkspaceRunSlot } from "./workspace-run-lock";

const originalStoreUrl = process.env.BYOK_SECRET_STORE_URL;
const originalStoreAuthToken = process.env.BYOK_SECRET_STORE_AUTH_TOKEN;

let tempDir: string;

describe("workspace-run-lock", () => {
  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), "workspace-run-lock-"));
    process.env.BYOK_SECRET_STORE_URL = `file:${join(tempDir, "control-store.db")}`;
    delete process.env.BYOK_SECRET_STORE_AUTH_TOKEN;
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    if (originalStoreUrl) {
      process.env.BYOK_SECRET_STORE_URL = originalStoreUrl;
    } else {
      delete process.env.BYOK_SECRET_STORE_URL;
    }
    if (originalStoreAuthToken) {
      process.env.BYOK_SECRET_STORE_AUTH_TOKEN = originalStoreAuthToken;
    } else {
      delete process.env.BYOK_SECRET_STORE_AUTH_TOKEN;
    }
    rmSync(tempDir, { recursive: true, force: true });
  });

  it("expires stale reservations and allows a fresh claim", async () => {
    vi.setSystemTime(new Date("2026-03-24T12:00:00Z"));

    const firstClaim = await claimWorkspaceRunSlot("workspace-1");
    if (firstClaim.status !== "claimed") {
      throw new Error("Expected claimed workspace slot");
    }

    vi.setSystemTime(new Date("2026-03-24T12:16:00Z"));

    const secondClaim = await claimWorkspaceRunSlot("workspace-1");
    if (secondClaim.status !== "claimed") {
      throw new Error("Expected claimed workspace slot");
    }

    expect(secondClaim.reservationId).not.toBe(firstClaim.reservationId);
  });

  it("releases the active reservation for immediate reuse", async () => {
    vi.setSystemTime(new Date("2026-03-24T12:00:00Z"));

    const claim = await claimWorkspaceRunSlot("workspace-2");
    expect(claim.status).toBe("claimed");

    if (claim.status !== "claimed") {
      throw new Error("Expected claimed workspace slot");
    }

    await releaseWorkspaceRunSlot("workspace-2", claim.reservationId);

    const nextClaim = await claimWorkspaceRunSlot("workspace-2");
    if (nextClaim.status !== "claimed") {
      throw new Error("Expected claimed workspace slot");
    }

    expect(nextClaim.reservationId).not.toBe(claim.reservationId);
  });
});
