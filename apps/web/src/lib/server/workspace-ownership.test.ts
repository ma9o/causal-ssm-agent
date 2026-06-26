import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const readdirMock = vi.hoisted(() => vi.fn());

vi.mock("node:fs/promises", () => ({
  readdir: readdirMock,
}));

vi.mock("@/lib/server/openrouter-access", () => ({
  resolveOpenRouterAccess: vi.fn(),
}));

vi.mock("@/lib/server/workspace-session", () => ({
  readAuthorizedWorkspaceIds: vi.fn(),
}));

vi.mock("@/lib/storage", () => ({
  LOCAL_DATA_DIR: "/tmp/data",
  isStorageNotFoundError: vi.fn(
    (error: unknown) => error instanceof Error && error.message.startsWith("ENOENT:"),
  ),
  prefixExists: vi.fn(),
  readData: vi.fn(),
}));

import { resolveOpenRouterAccess } from "@/lib/server/openrouter-access";
import { readAuthorizedWorkspaceIds } from "@/lib/server/workspace-session";
import { prefixExists, readData } from "@/lib/storage";
import {
  authorizeWorkspaceForOpenRouterUser,
  authorizeWorkspacesForOpenRouterUser,
  listAccessibleWorkspaces,
  readOpenRouterOwnedWorkspaceIds,
} from "./workspace-ownership";

const originalStoreUrl = process.env.BYOK_SECRET_STORE_URL;
const originalStoreAuthToken = process.env.BYOK_SECRET_STORE_AUTH_TOKEN;

describe("workspace-ownership", () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), "workspace-ownership-"));
    process.env.BYOK_SECRET_STORE_URL = `file:${join(tempDir, "control-store.db")}`;
    delete process.env.BYOK_SECRET_STORE_AUTH_TOKEN;

    vi.mocked(readData).mockImplementation(async (path: string) => {
      if (path === "OWNED1/query.txt") {
        return "Owned question";
      }
      if (path === "DEFAULT/query.txt") {
        return "Shared question";
      }
      throw new Error(`ENOENT: ${path}`);
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    if (originalStoreUrl === undefined) {
      delete process.env.BYOK_SECRET_STORE_URL;
    } else {
      process.env.BYOK_SECRET_STORE_URL = originalStoreUrl;
    }
    if (originalStoreAuthToken === undefined) {
      delete process.env.BYOK_SECRET_STORE_AUTH_TOKEN;
    } else {
      process.env.BYOK_SECRET_STORE_AUTH_TOKEN = originalStoreAuthToken;
    }
    rmSync(tempDir, { recursive: true, force: true });
  });

  it("lists every local workspace under ./data in local mode", async () => {
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "local",
      apiKey: "local-key",
    });
    readdirMock.mockResolvedValue([
      { isDirectory: () => true, name: ".private" },
      { isDirectory: () => true, name: "LOCAL_A" },
      { isDirectory: () => false, name: "notes.txt" },
      { isDirectory: () => true, name: "LOCAL_B" },
    ]);

    const workspaces = await listAccessibleWorkspaces();

    expect(workspaces).toEqual({
      mode: "local",
      workspaces: [
        {
          href: "/analysis/LOCAL_A",
          question: null,
          source: "local",
          workspaceId: "LOCAL_A",
        },
        {
          href: "/analysis/LOCAL_B",
          question: null,
          source: "local",
          workspaceId: "LOCAL_B",
        },
      ],
    });
  });

  it("lists only account-owned and shared workspaces in user mode", async () => {
    vi.mocked(resolveOpenRouterAccess).mockResolvedValue({
      mode: "user",
      apiKey: "user-key",
      userId: "or-user-123",
    });
    vi.mocked(readAuthorizedWorkspaceIds).mockResolvedValue(["OWNED1", "SESSIONONLY"]);
    await authorizeWorkspacesForOpenRouterUser("or-user-123", ["OWNED1", "OWNED2"]);
    vi.mocked(prefixExists).mockImplementation(
      async (path: string) => path === "OWNED1/" || path === "OWNED2/" || path === "DEFAULT/",
    );

    const workspaces = await listAccessibleWorkspaces();

    expect(workspaces).toEqual({
      mode: "user",
      workspaces: [
        {
          href: "/analysis/OWNED1",
          question: "Owned question",
          source: "user",
          workspaceId: "OWNED1",
        },
        {
          href: "/analysis/OWNED2",
          question: null,
          source: "user",
          workspaceId: "OWNED2",
        },
        {
          href: "/analysis/DEFAULT",
          question: "Shared question",
          source: "shared",
          workspaceId: "DEFAULT",
        },
      ],
    });
  });

  it("keeps distinct workspaces when the same user authorizes them concurrently", async () => {
    await Promise.all([
      authorizeWorkspaceForOpenRouterUser("or-user-123", "OWNED1"),
      authorizeWorkspaceForOpenRouterUser("or-user-123", "OWNED2"),
    ]);

    const workspaceIds = await readOpenRouterOwnedWorkspaceIds("or-user-123");

    expect(new Set(workspaceIds)).toEqual(new Set(["OWNED1", "OWNED2"]));
  });

  it("keeps owned workspaces even if their artifacts are missing", async () => {
    await authorizeWorkspacesForOpenRouterUser("or-user-123", ["LIVE1", "MISSING1"]);

    await expect(readOpenRouterOwnedWorkspaceIds("or-user-123")).resolves.toEqual([
      "LIVE1",
      "MISSING1",
    ]);
  });
});
