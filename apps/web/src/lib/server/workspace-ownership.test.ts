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
  prefixExists: vi.fn(),
  readData: vi.fn(),
  writeData: vi.fn(),
}));

import { resolveOpenRouterAccess } from "@/lib/server/openrouter-access";
import { readAuthorizedWorkspaceIds } from "@/lib/server/workspace-session";
import { prefixExists, readData, writeData } from "@/lib/storage";
import {
  authorizeWorkspaceForOpenRouterUser,
  listAccessibleWorkspaces,
  readOpenRouterOwnedWorkspaceIds,
} from "./workspace-ownership";

describe("workspace-ownership", () => {
  beforeEach(() => {
    vi.mocked(readData).mockImplementation(async (path: string) => {
      if (path.includes("/workspaces.json")) {
        return JSON.stringify({ workspaceIds: ["OWNED1", "OWNED2"] });
      }
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
    vi.mocked(prefixExists).mockImplementation(
      async (path: string) =>
        path === "OWNED1/" || path === "OWNED2/" || path === "DEFAULT/",
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

  it("persists user-owned workspaces in recency order", async () => {
    vi.mocked(readData).mockResolvedValueOnce(
      JSON.stringify({ workspaceIds: ["OWNED1", "OWNED2"] }),
    );

    await authorizeWorkspaceForOpenRouterUser("or-user-123", "OWNED2");

    expect(writeData).toHaveBeenCalledWith(
      expect.stringContaining("/workspaces.json"),
      JSON.stringify({ workspaceIds: ["OWNED2", "OWNED1"] }, null, 2),
    );
    await expect(readOpenRouterOwnedWorkspaceIds("or-user-123")).resolves.toEqual([
      "OWNED1",
      "OWNED2",
    ]);
  });
});
