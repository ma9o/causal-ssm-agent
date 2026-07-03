import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/storage", () => ({
  ensureDir: vi.fn(),
  writeBinary: vi.fn(),
}));

import { ensureDir, writeBinary } from "@/lib/storage";
import { POST } from "./route";

describe("POST /api/upload", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("stores the uploaded file under the workspace input prefix", async () => {
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
  });

  it("rejects empty file names before writing", async () => {
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
  });

  it("rejects malformed workspace ids", async () => {
    const file = new File(["hello"], "data.csv", { type: "text/csv" });
    const formData = new FormData();
    formData.set("workspaceId", "../etc");
    formData.set("file", file);

    const response = await POST(
      new Request("http://localhost/api/upload", {
        method: "POST",
        body: formData,
      }),
    );

    expect(response.status).toBe(400);
    expect(writeBinary).not.toHaveBeenCalled();
  });
});
