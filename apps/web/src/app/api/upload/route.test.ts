import { afterEach, describe, expect, it, vi } from "vitest";
import { POST } from "./route";

describe("POST /api/upload", () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("proxies the uploaded file to the facade upload endpoint", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json({
          path: "NEWSPACE/input/data.csv",
        }),
      ),
    );

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

    const [url, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:8100/api/upload");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("workspaceId")).toBe("NEWSPACE");
    const proxiedFile = (init.body as FormData).get("file");
    expect(proxiedFile).toBeInstanceOf(File);
    expect((proxiedFile as File).name).toBe("data.csv");
    expect((proxiedFile as File).size).toBe(file.size);
    expect((proxiedFile as File).type).toBe("text/csv");
  });

  it("rejects empty file names before proxying", async () => {
    vi.stubGlobal("fetch", vi.fn());
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
    expect(fetch).not.toHaveBeenCalled();
  });

  it("rejects malformed workspace ids", async () => {
    vi.stubGlobal("fetch", vi.fn());
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
    expect(fetch).not.toHaveBeenCalled();
  });
});
