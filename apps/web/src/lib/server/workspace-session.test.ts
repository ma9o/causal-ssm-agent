import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const cookiesMock = vi.hoisted(() => vi.fn());
const getIronSessionMock = vi.hoisted(() => vi.fn());

vi.mock("next/headers", () => ({
  cookies: cookiesMock,
}));

vi.mock("iron-session", () => ({
  getIronSession: getIronSessionMock,
}));

import {
  authorizeWorkspaceInSession,
  clearAuthorizedWorkspaceIds,
  hasWorkspaceSessionAccess,
  readAuthorizedWorkspaceIds,
  replaceAuthorizedWorkspaceIds,
} from "./workspace-session";

type MockWorkspaceSessionStore = {
  save: ReturnType<typeof vi.fn>;
  workspaceIds?: string[];
};

const originalAppSecret = process.env.APP_SECRET;

describe("workspace-session", () => {
  let session: MockWorkspaceSessionStore;

  beforeEach(() => {
    process.env.APP_SECRET = "0123456789abcdef0123456789abcdef";
    session = { save: vi.fn() };
    cookiesMock.mockResolvedValue({});
    getIronSessionMock.mockResolvedValue(session);
  });

  afterEach(() => {
    vi.clearAllMocks();
    if (originalAppSecret === undefined) {
      delete process.env.APP_SECRET;
    } else {
      process.env.APP_SECRET = originalAppSecret;
    }
  });

  it("reads and normalizes workspace ids from the session store", async () => {
    session.workspaceIds = ["WS1", " WS2 ", "WS1", "", "WS3"];

    await expect(readAuthorizedWorkspaceIds()).resolves.toEqual(["WS1", "WS2", "WS3"]);
    await expect(hasWorkspaceSessionAccess("WS2")).resolves.toBe(true);
  });

  it("moves the most recent workspace to the front and persists the session", async () => {
    session.workspaceIds = ["WS1", "WS2"];

    await authorizeWorkspaceInSession("WS2");

    expect(session.workspaceIds).toEqual(["WS2", "WS1"]);
    expect(session.save).toHaveBeenCalledTimes(1);
  });

  it("replaces and clears the authorized workspace list", async () => {
    await replaceAuthorizedWorkspaceIds(["NEW1", "NEW2", "NEW1"]);
    expect(session.workspaceIds).toEqual(["NEW1", "NEW2"]);

    await clearAuthorizedWorkspaceIds();
    expect(session.workspaceIds).toEqual([]);
    expect(session.save).toHaveBeenCalledTimes(2);
  });
});
