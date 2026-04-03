import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/storage", () => ({
  ensureDir: vi.fn(),
  writeBinary: vi.fn(),
}));

vi.mock("@/lib/workspace-access", () => ({
  finalizeWorkspaceCreate: vi.fn(),
  requireWorkspaceAccess: vi.fn(),
}));

import { ensureDir, writeBinary } from "@/lib/storage";
import { finalizeWorkspaceCreate, requireWorkspaceAccess } from "@/lib/workspace-access";
import { POST } from "./route";

describe("POST /api/upload", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("finalizes a new workspace only after the upload succeeds", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "NEWSPACE",
      creationPending: true,
    });

    const file = new File(["hello"], "data.csv", { type: "text/csv" });
    const formData = new FormData();
    formData.set("workspaceId", "NEWSPACE");
    formData.set("file", file);

    const response = await POST(
      new Request("http://localhost/api/upload", {
        method: "POST",
        body: formData,
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      path: "NEWSPACE/input/data.csv",
    });
    expect(ensureDir).toHaveBeenCalledWith("NEWSPACE/input");
    expect(writeBinary).toHaveBeenCalledTimes(1);
    expect(finalizeWorkspaceCreate).toHaveBeenCalledWith("NEWSPACE");
  });

  it("does not finalize a workspace when validation fails before the write", async () => {
    vi.mocked(requireWorkspaceAccess).mockResolvedValue({
      ok: true,
      workspaceId: "BROKEN",
      creationPending: true,
    });

    const file = new File(["hello"], "", { type: "text/csv" });
    const formData = new FormData();
    formData.set("workspaceId", "BROKEN");
    formData.set("file", file);

    const response = await POST(
      new Request("http://localhost/api/upload", {
        method: "POST",
        body: formData,
      }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({ error: "Invalid file name" });
    expect(writeBinary).not.toHaveBeenCalled();
    expect(finalizeWorkspaceCreate).not.toHaveBeenCalled();
  });
});
