import { afterEach, describe, expect, it, vi } from "vitest";
import type { MoveOutcome } from "@/lib/server/episode-runs";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi
    .fn()
    .mockImplementation(async (_request: Request, workspaceId: string) => ({
      ok: true,
      workspaceId,
    })),
}));

vi.mock("@/lib/server/episode-runs", () => ({
  proposeMove: vi.fn(),
}));

vi.mock("@/lib/storage", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/storage")>();
  return {
    ...actual,
    readData: vi.fn(),
  };
});

import { proposeMove } from "@/lib/server/episode-runs";
import { readData } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { POST } from "./route";

function appliedOutcome(): MoveOutcome {
  return {
    seq: 12,
    status: "applied",
    reason: null,
    error_type: null,
    error_message: null,
    diagnostics: {},
    produced: [],
    retracted: [],
    state: { current: {} },
  };
}

describe("POST /api/refine/apply", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it("persists terminal Stage 6 results as a human write move", async () => {
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
    vi.mocked(proposeMove).mockResolvedValue(appliedOutcome());

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
    expect(proposeMove).toHaveBeenCalledTimes(1);
    expect(proposeMove).toHaveBeenCalledWith(
      "user-123",
      { kind: "write", artifact_id: "baseline_ranking", provenance: "human" },
      {
        intervention_results: [{ treatment: "Stress", effect_size: -0.4, identifiable: true }],
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
              content: "Preserve the statin-adherence scenario and compare it against rung 3 next.",
              tool_is_error: false,
            },
          ],
        },
      },
    );
    expect(requireWorkspaceAccess).toHaveBeenCalledWith(expect.any(Request), "user-123", {
      requireMutable: true,
    });
  });

  it("surfaces failed terminal writes as 502", async () => {
    vi.mocked(readData).mockResolvedValueOnce(JSON.stringify({ intervention_results: [] }));
    vi.mocked(proposeMove).mockResolvedValue({
      ...appliedOutcome(),
      status: "raised",
      error_type: "SchemaValidationError",
      error_message: "baseline_ranking payload failed validation",
    });

    const response = await POST(
      new Request("http://localhost/api/refine/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-6",
          stagePatch: { final_summary: "Updated." },
          messages: [],
        }),
      }),
    );

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      error: "Persist failed: baseline_ranking payload failed validation",
    });
  });

  it("replays non-terminal stages from the client-held materialization payload", async () => {
    vi.mocked(readData).mockResolvedValueOnce(
      JSON.stringify({
        latent_model: { constructs: [{ name: "Old" }], edges: [] },
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
          workspaceId: "user-123",
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
      workspaceId: "user-123",
    });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [, init] = fetchSpy.mock.calls[0] ?? [];
    const headers = (init as RequestInit | undefined)?.headers as Headers | undefined;
    expect(fetchSpy).toHaveBeenCalledWith(
      new URL("/api/replay", "http://localhost/api/refine/apply"),
      expect.objectContaining({
        method: "POST",
      }),
    );
    expect(headers).toBeInstanceOf(Headers);
    expect(headers?.get("Content-Type")).toBe("application/json");
    const body = JSON.parse(String((init as RequestInit | undefined)?.body ?? "{}"));
    expect(body).toMatchObject({
      workspaceId: "user-123",
      stageId: "stage-1a",
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
    expect(body).not.toHaveProperty("rootFlowRunId");
  });

  it("forwards the workspace access cookie to the replay route", async () => {
    vi.mocked(readData).mockResolvedValueOnce(
      JSON.stringify({
        causal_spec: { measurement: { indicators: [] } },
      }),
    );

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true, workspaceId: "user-123" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const response = await POST(
      new Request("http://localhost/api/refine/apply", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          cookie: "workspace_session=test-cookie",
        },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-1b",
          stagePatch: {
            causal_spec: { measurement: { indicators: [] } },
          },
          messages: [],
        }),
      }),
    );

    expect(response.status).toBe(200);
    const [, init] = fetchSpy.mock.calls[0] ?? [];
    const headers = (init as RequestInit | undefined)?.headers as Headers | Record<string, string>;

    expect(fetchSpy).toHaveBeenCalledWith(
      new URL("/api/replay", "http://localhost/api/refine/apply"),
      expect.objectContaining({
        method: "POST",
      }),
    );
    expect(headers).toBeInstanceOf(Headers);
    expect((headers as Headers).get("cookie")).toBe("workspace_session=test-cookie");
  });
});
