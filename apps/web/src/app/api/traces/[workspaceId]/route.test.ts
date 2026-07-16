import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/episode-runs", () => ({
  EpisodeRunError: class EpisodeRunError extends Error {
    constructor(
      public status: number,
      message: string,
    ) {
      super(message);
    }
  },
  getArtifactTraceIndex: vi.fn(),
  getEpisodeTrace: vi.fn(),
}));

import { getArtifactTraceIndex, getEpisodeTrace } from "@/lib/server/episode-runs";
import { GET } from "./route";

describe("GET /api/traces/[workspaceId]", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("resolves the producing transition and merges its promoted traces", async () => {
    vi.mocked(getArtifactTraceIndex).mockResolvedValue({
      workspace_id: "DEMO",
      artifact_id: "statistical_model_spec",
      version: 1,
      seq: 6,
      trace_ids: ["construct-a", "construct-b"],
    });
    vi.mocked(getEpisodeTrace)
      .mockResolvedValueOnce({
        messages: [{ role: "assistant", content: "A", tool_is_error: false }],
        model: "model-a",
        total_time_seconds: 1,
        usage: { input_tokens: 2, output_tokens: 3, reasoning_tokens: null },
      })
      .mockResolvedValueOnce({
        messages: [{ role: "assistant", content: "B", tool_is_error: false }],
        model: "model-b",
        total_time_seconds: 4,
        usage: { input_tokens: 5, output_tokens: 6, reasoning_tokens: 7 },
      });

    const response = await GET(
      new Request("http://localhost/api/traces/DEMO?artifact=statistical_model_spec"),
      { params: Promise.resolve({ workspaceId: "DEMO" }) },
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      messages: [
        { role: "assistant", content: "A", tool_is_error: false },
        { role: "assistant", content: "B", tool_is_error: false },
      ],
      model: "model-b",
      total_time_seconds: 5,
      usage: { input_tokens: 7, output_tokens: 9, reasoning_tokens: 7 },
    });
    expect(getEpisodeTrace).toHaveBeenNthCalledWith(1, "DEMO", 6, "construct-a");
    expect(getEpisodeTrace).toHaveBeenNthCalledWith(2, "DEMO", 6, "construct-b");
  });

  it("returns 404 when the producing transition has no promoted traces", async () => {
    vi.mocked(getArtifactTraceIndex).mockResolvedValue({
      workspace_id: "DEMO",
      artifact_id: "posterior",
      version: 1,
      seq: 8,
      trace_ids: [],
    });

    const response = await GET(new Request("http://localhost/api/traces/DEMO?artifact=posterior"), {
      params: Promise.resolve({ workspaceId: "DEMO" }),
    });

    expect(response.status).toBe(404);
  });
});
