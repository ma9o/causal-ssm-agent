import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi.fn().mockImplementation(async (_request: Request, workspaceId: string) => ({
    ok: true,
    workspaceId,
  })),
}));

vi.mock("@/lib/runtime-urls", () => ({
  getToolServerUrl: vi.fn(() => "http://tools.example"),
}));

vi.mock("@/lib/storage", () => ({
  readData: vi.fn(),
}));

import { readData } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { POST } from "./route";

describe("POST /api/refine/apply", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it("persists terminal Stage 6 results from the client-held patch and messages", async () => {
    vi.mocked(readData).mockResolvedValueOnce(
      JSON.stringify({
        intervention_results: [{ treatment: "Stress", effect_size: -0.4, identifiable: true }],
        llm_trace: {
          model: "openrouter/anthropic/claude-sonnet-4",
          total_time_seconds: 5,
          usage: { input_tokens: 10, output_tokens: 5 },
          messages: [],
        },
      }),
    );

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({ ok: true }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    const response = await POST(
      new Request("http://localhost/api/refine/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-6",
          stagePatch: {
            final_summary: "Statin adherence remains the dominant lever.",
          },
          messages: [
            {
              id: "user-1",
              role: "user",
              parts: [{ type: "text", text: "What should I preserve?" }],
            },
            {
              id: "assistant-1",
              role: "assistant",
              metadata: {
                durationSeconds: 1.5,
                stagePatch: {
                  saved_scenarios: [{ label: "High adherence", query: "rung2" }],
                },
                usage: {
                  inputTokens: 11,
                  outputTokens: 7,
                  reasoningTokens: 3,
                },
              },
              parts: [
                {
                  type: "text",
                  text: "Preserve the statin-adherence scenario and compare it against rung 3 next.",
                },
              ],
            },
          ],
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      updatedFields: ["saved_scenarios", "final_summary", "llm_trace"],
    });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy).toHaveBeenCalledWith(
      "http://tools.example/api/stages/stage-6/persist-web-patch",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }),
    );
    const [, init] = fetchSpy.mock.calls[0] ?? [];
    expect(JSON.parse(String((init as RequestInit | undefined)?.body ?? "{}"))).toEqual({
      workspace_id: "user-123",
      patch: {
        saved_scenarios: [{ label: "High adherence", query: "rung2" }],
        final_summary: "Statin adherence remains the dominant lever.",
        llm_trace: {
          model: "openrouter/anthropic/claude-sonnet-4",
          total_time_seconds: 6.5,
          usage: { input_tokens: 21, output_tokens: 12, reasoning_tokens: 3 },
          messages: [
            {
              role: "user",
              content: "What should I preserve?",
              tool_is_error: false,
            },
            {
              role: "assistant",
              content:
                "Preserve the statin-adherence scenario and compare it against rung 3 next.",
              tool_is_error: false,
            },
          ],
        },
      },
    });
    expect(requireWorkspaceAccess).toHaveBeenCalledWith(expect.any(Request), "user-123");
  });

  it("replays non-terminal stages from the client-held materialization payload", async () => {
    vi.mocked(readData).mockResolvedValueOnce(
      JSON.stringify({
        latent_model: { constructs: [{ name: "Old" }], edges: [] },
        outcome: "success",
        llm_trace: {
          model: "openrouter/anthropic/claude-sonnet-4",
          total_time_seconds: 7,
          usage: { input_tokens: 12, output_tokens: 9 },
          messages: [{ role: "assistant", content: "Refined trace", tool_is_error: false }],
        },
      }),
    );

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          ok: true,
          resumeFrom: "stage-1b",
          rootFlowRunId: "replay-1",
          sessionPersisted: true,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    const response = await POST(
      new Request("http://localhost/api/refine/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-1a",
          rootFlowRunId: "root-123",
          stagePatch: {},
          messages: [
            {
              id: "user-1",
              role: "user",
              parts: [{ type: "text", text: "Tighten the latent model." }],
            },
            {
              id: "assistant-1",
              role: "assistant",
              metadata: {
                durationSeconds: 1.25,
                stagePatch: {
                  latent_model: { constructs: [], edges: [] },
                },
                usage: {
                  inputTokens: 5,
                  outputTokens: 4,
                  reasoningTokens: 1,
                },
              },
              parts: [{ type: "text", text: "The latent model is valid." }],
            },
          ],
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      updatedFields: ["latent_model", "llm_trace"],
      resumeFrom: "stage-1b",
      rootFlowRunId: "replay-1",
      sessionPersisted: true,
    });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy).toHaveBeenCalledWith(
      new URL("/api/replay", "http://localhost/api/refine/apply"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }),
    );
    const [, init] = fetchSpy.mock.calls[0] ?? [];
    const body = JSON.parse(String((init as RequestInit | undefined)?.body ?? "{}"));
    expect(body).toMatchObject({
      workspaceId: "user-123",
      stageId: "stage-1a",
      rootFlowRunId: "root-123",
      stageData: {
        latent_model: { constructs: [], edges: [] },
        llm_trace: {
          model: "openrouter/anthropic/claude-sonnet-4",
          total_time_seconds: 8.25,
          usage: { input_tokens: 17, output_tokens: 13, reasoning_tokens: 1 },
          messages: [
            { role: "assistant", content: "Refined trace", tool_is_error: false },
            { role: "user", content: "Tighten the latent model.", tool_is_error: false },
            { role: "assistant", content: "The latent model is valid.", tool_is_error: false },
          ],
        },
      },
    });
  });
});
