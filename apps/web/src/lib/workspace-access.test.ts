import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/workspace-session", () => ({
  authorizeWorkspaceInSession: vi.fn(),
  hasWorkspaceSessionAccess: vi.fn(),
}));

vi.mock("@/lib/server/workspace-ownership", () => ({
  authorizeWorkspaceForOpenRouterUser: vi.fn(),
  hasOpenRouterWorkspaceAccess: vi.fn(),
  resolveWorkspaceOwnershipContext: vi.fn(),
}));

vi.mock("@/lib/storage", () => ({
  prefixExists: vi.fn(),
}));

import {
  authorizeWorkspaceForOpenRouterUser,
  hasOpenRouterWorkspaceAccess,
  resolveWorkspaceOwnershipContext,
} from "@/lib/server/workspace-ownership";
import {
  authorizeWorkspaceInSession,
  hasWorkspaceSessionAccess,
} from "@/lib/server/workspace-session";
import { prefixExists } from "@/lib/storage";
import { finalizeWorkspaceCreate, requireWorkspaceAccess } from "./workspace-access";

describe("requireWorkspaceAccess", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("rejects invalid workspace ids", async () => {
    const result = await requireWorkspaceAccess(new Request("http://localhost"), "bad/workspace");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.response.status).toBe(400);
    }
  });

  it("allows shared fixture workspaces without a session", async () => {
    const result = await requireWorkspaceAccess(new Request("http://localhost"), "DEFAULT");

    expect(result).toEqual({
      ok: true,
      workspaceId: "DEFAULT",
      creationPending: false,
    });
    expect(hasWorkspaceSessionAccess).not.toHaveBeenCalled();
  });

  it("routes the demo-health shared fixture alias to DEMO", async () => {
    const result = await requireWorkspaceAccess(new Request("http://localhost"), "demo_health");

    expect(result).toEqual({
      ok: true,
      workspaceId: "DEMO",
      creationPending: false,
    });
    expect(hasWorkspaceSessionAccess).not.toHaveBeenCalled();
  });

  it("refuses create access on shared fixture workspaces", async () => {
    const result = await requireWorkspaceAccess(new Request("http://localhost"), "DEFAULT", {
      allowCreate: true,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.response.status).toBe(403);
    }
  });

  it("allows workspaces already present in the browser session", async () => {
    vi.mocked(resolveWorkspaceOwnershipContext).mockResolvedValue({ mode: "anonymous" });
    vi.mocked(hasWorkspaceSessionAccess).mockResolvedValue(true);

    const result = await requireWorkspaceAccess(new Request("http://localhost"), "USER123");

    expect(result).toEqual({
      ok: true,
      workspaceId: "USER123",
      creationPending: false,
    });
    expect(prefixExists).not.toHaveBeenCalled();
  });

  it("marks a fresh anonymous workspace for creation without persisting it yet", async () => {
    vi.mocked(resolveWorkspaceOwnershipContext).mockResolvedValue({ mode: "anonymous" });
    vi.mocked(hasWorkspaceSessionAccess).mockResolvedValue(false);
    vi.mocked(prefixExists).mockResolvedValue(false);

    const result = await requireWorkspaceAccess(new Request("http://localhost"), "NEWSPACE", {
      allowCreate: true,
    });

    expect(result).toEqual({
      ok: true,
      workspaceId: "NEWSPACE",
      creationPending: true,
    });
    expect(prefixExists).toHaveBeenCalledWith("NEWSPACE/");
    expect(authorizeWorkspaceInSession).not.toHaveBeenCalled();
  });

  it("refuses to claim an existing workspace for a new browser session", async () => {
    vi.mocked(resolveWorkspaceOwnershipContext).mockResolvedValue({ mode: "anonymous" });
    vi.mocked(hasWorkspaceSessionAccess).mockResolvedValue(false);
    vi.mocked(prefixExists).mockResolvedValue(true);

    const result = await requireWorkspaceAccess(new Request("http://localhost"), "USER123", {
      allowCreate: true,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.response.status).toBe(403);
    }
    expect(authorizeWorkspaceInSession).not.toHaveBeenCalled();
  });

  it("allows a user-owned workspace without relying on the browser session", async () => {
    vi.mocked(resolveWorkspaceOwnershipContext).mockResolvedValue({
      mode: "user",
      userId: "or-user-123",
    });
    vi.mocked(hasOpenRouterWorkspaceAccess).mockResolvedValue(true);

    const result = await requireWorkspaceAccess(new Request("http://localhost"), "OWNED123");

    expect(result).toEqual({
      ok: true,
      workspaceId: "OWNED123",
      creationPending: false,
    });
    expect(hasWorkspaceSessionAccess).not.toHaveBeenCalled();
  });

  it("marks a newly created user workspace without leaking it into the session", async () => {
    vi.mocked(resolveWorkspaceOwnershipContext).mockResolvedValue({
      mode: "user",
      userId: "or-user-123",
    });
    vi.mocked(hasOpenRouterWorkspaceAccess).mockResolvedValue(false);
    vi.mocked(prefixExists).mockResolvedValue(false);

    const result = await requireWorkspaceAccess(new Request("http://localhost"), "USERSPACE", {
      allowCreate: true,
    });

    expect(result).toEqual({
      ok: true,
      workspaceId: "USERSPACE",
      creationPending: true,
    });
    expect(hasWorkspaceSessionAccess).not.toHaveBeenCalled();
    expect(authorizeWorkspaceForOpenRouterUser).not.toHaveBeenCalled();
    expect(authorizeWorkspaceInSession).not.toHaveBeenCalled();
  });

  it("allows existing workspaces directly from local ownership mode", async () => {
    vi.mocked(resolveWorkspaceOwnershipContext).mockResolvedValue({ mode: "local" });
    vi.mocked(prefixExists).mockResolvedValue(true);

    const result = await requireWorkspaceAccess(new Request("http://localhost"), "LOCAL123");

    expect(result).toEqual({
      ok: true,
      workspaceId: "LOCAL123",
      creationPending: false,
    });
    expect(hasWorkspaceSessionAccess).not.toHaveBeenCalled();
  });

  it("finalizes anonymous workspace creation into the browser session", async () => {
    vi.mocked(resolveWorkspaceOwnershipContext).mockResolvedValue({ mode: "anonymous" });

    await finalizeWorkspaceCreate("ANON123");

    expect(authorizeWorkspaceInSession).toHaveBeenCalledWith("ANON123");
    expect(authorizeWorkspaceForOpenRouterUser).not.toHaveBeenCalled();
  });

  it("finalizes user workspace creation into the OpenRouter-owned index only", async () => {
    vi.mocked(resolveWorkspaceOwnershipContext).mockResolvedValue({
      mode: "user",
      userId: "or-user-123",
    });

    await finalizeWorkspaceCreate("USER123");

    expect(authorizeWorkspaceForOpenRouterUser).toHaveBeenCalledWith("or-user-123", "USER123");
    expect(authorizeWorkspaceInSession).not.toHaveBeenCalled();
  });
});
