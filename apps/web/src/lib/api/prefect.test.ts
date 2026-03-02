import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("./client", () => ({
  apiFetch: vi.fn(),
}));

import { apiFetch } from "./client";
import { getDeploymentId, triggerRun } from "./prefect";

describe("getDeploymentId", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns deployment id when found", async () => {
    vi.mocked(apiFetch).mockResolvedValue([{ id: "dep-123", name: "causal-inference" }]);

    const id = await getDeploymentId();
    expect(id).toBe("dep-123");
  });

  it("uses default name 'causal-inference'", async () => {
    vi.mocked(apiFetch).mockResolvedValue([{ id: "dep-123", name: "causal-inference" }]);

    await getDeploymentId();

    expect(apiFetch).toHaveBeenCalledWith(
      "/prefect/deployments/filter",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining("causal-inference"),
      }),
    );
  });

  it("accepts custom deployment name", async () => {
    vi.mocked(apiFetch).mockResolvedValue([{ id: "dep-456", name: "custom" }]);

    const id = await getDeploymentId("custom");
    expect(id).toBe("dep-456");
    expect(apiFetch).toHaveBeenCalledWith(
      "/prefect/deployments/filter",
      expect.objectContaining({
        body: expect.stringContaining("custom"),
      }),
    );
  });

  it("throws when no deployment found", async () => {
    vi.mocked(apiFetch).mockResolvedValue([]);

    await expect(getDeploymentId("missing")).rejects.toThrow('Deployment "missing" not found');
  });

  it("returns first deployment when multiple match", async () => {
    vi.mocked(apiFetch).mockResolvedValue([
      { id: "first", name: "causal-inference" },
      { id: "second", name: "causal-inference" },
    ]);

    const id = await getDeploymentId();
    expect(id).toBe("first");
  });
});

describe("triggerRun", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns flow run id", async () => {
    vi.mocked(apiFetch).mockResolvedValue({
      id: "run-789",
      state: { type: "SCHEDULED", name: "Scheduled" },
    });

    const id = await triggerRun("dep-123", { user_id: "test" });
    expect(id).toBe("run-789");
  });

  it("sends parameters in request body", async () => {
    vi.mocked(apiFetch).mockResolvedValue({
      id: "run-1",
      state: { type: "SCHEDULED", name: "Scheduled" },
    });

    await triggerRun("dep-123", { user_id: "test", query: "test.txt" });

    expect(apiFetch).toHaveBeenCalledWith(
      "/prefect/deployments/dep-123/create_flow_run",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ parameters: { user_id: "test", query: "test.txt" } }),
      }),
    );
  });

  it("propagates API errors from apiFetch", async () => {
    vi.mocked(apiFetch).mockRejectedValue(new Error("API error 500: Internal Server Error"));

    await expect(triggerRun("dep-123", { user_id: "test" })).rejects.toThrow("API error 500");
  });
});
