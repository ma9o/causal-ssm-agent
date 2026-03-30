import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { baseTrace, streamTextMock } = vi.hoisted(() => ({
  baseTrace: {
    model: "openrouter/anthropic/claude-sonnet-4",
    total_time_seconds: 2.5,
    usage: {
      input_tokens: 100,
      output_tokens: 40,
      reasoning_tokens: 8,
    },
    messages: [
      {
        role: "system",
        content: "Comment on the Stage 6 results.",
        tool_is_error: false,
      },
      {
        role: "assistant",
        content: "Initial stage commentary.",
        tool_is_error: false,
      },
    ],
  },
  streamTextMock: vi.fn(),
}));

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi.fn().mockImplementation(async (_request: Request, workspaceId: string) => ({
    ok: true,
    workspaceId,
  })),
}));

vi.mock("@/lib/server/openrouter-access", () => ({
  resolveOpenRouterAccess: vi.fn().mockResolvedValue({
    mode: "user",
    apiKey: "test-key",
  }),
}));

vi.mock("@/lib/runtime-urls", () => ({
  getToolServerUrl: vi.fn(() => "http://tools.example"),
}));

vi.mock("@/lib/storage", () => ({
  readData: vi.fn().mockResolvedValue(JSON.stringify({ llm_trace: baseTrace })),
}));

vi.mock("@openrouter/ai-sdk-provider", () => ({
  createOpenRouter: vi.fn(() => vi.fn((model: string) => `model:${model}`)),
}));

vi.mock("ai", () => ({
  convertToModelMessages: vi.fn(async (messages) => messages),
  jsonSchema: vi.fn((schema) => schema),
  stepCountIs: vi.fn((count: number) => ({ type: "stepCountIs", count })),
  tool: vi.fn((definition) => definition),
  streamText: streamTextMock,
}));

import { readData } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { POST } from "./route";

describe("POST /api/refine", () => {
  beforeEach(() => {
    vi.mocked(readData).mockImplementation(async (path: string) => {
      if (path.endsWith("/stage-6.json") || path.endsWith("/stage-1a.json")) {
        return JSON.stringify({ llm_trace: baseTrace });
      }
      throw new Error(`ENOENT: ${path}`);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  it("emits usage metadata for stage-6 refinement turns", async () => {
    streamTextMock.mockImplementation(() => ({
      toUIMessageStreamResponse: ({
        messageMetadata,
        originalMessages,
      }: {
        messageMetadata?: (event: { part: unknown }) => unknown;
        originalMessages?: unknown[];
      }) =>
        new Response(
          JSON.stringify({
            originalMessages,
            metadata: messageMetadata?.({
              part: {
                type: "finish",
                totalUsage: {
                  inputTokens: 11,
                  outputTokens: 7,
                  outputTokenDetails: { reasoningTokens: 3 },
                  reasoningTokens: 3,
                },
              },
            }),
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
    }));

    const response = await POST(
      new Request("http://localhost/api/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-6",
          messages: [
            {
              id: "user-1",
              role: "user",
              parts: [{ type: "text", text: "What should I look at next?" }],
            },
          ],
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({
      originalMessages: [
        {
          id: "user-1",
          role: "user",
        },
      ],
      metadata: {
        stagePatch: {},
        usage: {
          inputTokens: 11,
          outputTokens: 7,
          reasoningTokens: 3,
        },
        durationSeconds: expect.any(Number),
      },
    });
    expect(readData).toHaveBeenCalledWith("user-123/run/stage-6.json");
    expect(requireWorkspaceAccess).toHaveBeenCalledWith(expect.any(Request), "user-123");
  });

  it("returns the merged stage patch to the client after tool execution", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          result: "VALID",
          stage_output: {
            latent_model: { constructs: [], edges: [] },
          },
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    streamTextMock.mockImplementation(({ tools }) => ({
      toUIMessageStreamResponse: async ({
        messageMetadata,
      }: {
        messageMetadata?: (event: { part: unknown }) => unknown;
      }) => {
        await tools.validate_latent_model.execute({
          structure_json: JSON.stringify({ constructs: [], edges: [] }),
        });

        return new Response(
          JSON.stringify({
            metadata: messageMetadata?.({
              part: {
                type: "finish",
                totalUsage: {
                  inputTokens: 5,
                  outputTokens: 4,
                  outputTokenDetails: { reasoningTokens: 1 },
                  reasoningTokens: 1,
                },
              },
            }),
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        );
      },
    }));

    const response = await POST(
      new Request("http://localhost/api/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-1a",
          pendingStagePatch: {
            reviewer_note: "carry forward",
          },
          messages: [
            {
              id: "user-1",
              role: "user",
              parts: [{ type: "text", text: "Tighten the latent model." }],
            },
          ],
        }),
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({
      metadata: {
        stagePatch: {
          reviewer_note: "carry forward",
          latent_model: { constructs: [], edges: [] },
        },
        usage: {
          inputTokens: 5,
          outputTokens: 4,
          reasoningTokens: 1,
        },
      },
    });

    expect(streamTextMock).toHaveBeenCalledWith(
      expect.objectContaining({
        stopWhen: { type: "stepCountIs", count: 10 },
      }),
    );
  });

  it("stringifies object payloads for tool params declared as strings", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          result: "VALID",
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    streamTextMock.mockImplementation(({ tools }) => ({
      toUIMessageStreamResponse: async () => {
        await tools.validate_measurement_model.execute({
          measurement_json: {
            model_clock: "1d",
            indicators: [],
          },
        });

        return new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      },
    }));

    vi.mocked(readData).mockResolvedValueOnce(JSON.stringify({ llm_trace: baseTrace }));

    const response = await POST(
      new Request("http://localhost/api/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-1b",
          messages: [
            {
              id: "user-1",
              role: "user",
              parts: [{ type: "text", text: "Validate the revised measurement model." }],
            },
          ],
        }),
      }),
    );

    expect(response.status).toBe(200);
    expect(fetchSpy).toHaveBeenCalledWith(
      "http://tools.example/api/tools/stage-1b/validate_measurement_model",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }),
    );
    const [, init] = fetchSpy.mock.calls[0] ?? [];
    expect(JSON.parse(String((init as RequestInit | undefined)?.body ?? "{}"))).toEqual({
      workspace_id: "user-123",
      input: {
        measurement_json: '{"model_clock":"1d","indicators":[]}',
      },
    });
  });

  it("rejects invalid message payloads", async () => {
    const response = await POST(
      new Request("http://localhost/api/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-6",
          messages: "not-an-array",
        }),
      }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({ error: "messages must be an array" });
  });

  it("appends broad Stage 4 refinement context after the saved trace", async () => {
    vi.mocked(readData).mockImplementation(async (path: string) => {
      if (path.endsWith("/stage-4.json")) {
        return JSON.stringify({
          llm_trace: baseTrace,
          outcome: "success",
          model_spec: {
            likelihoods: [
              {
                variable: "pss_score",
                distribution: "gaussian",
                link: "identity",
                reasoning: "Continuous score.",
                sources: [],
              },
            ],
            parameters: [
              {
                name: "beta_stress_sleep",
                role: "fixed_effect",
                constraint: "none",
                description: "Effect of stress on sleep.",
              },
            ],
          },
          authored_priors: {
            beta_stress_sleep: {
              parameter: "beta_stress_sleep",
              distribution: "Normal",
              params: { mu: -0.2, sigma: 0.1 },
              sources: [],
              reasoning: "Prior from longitudinal literature.",
            },
          },
          resolved_priors: [],
          search_queries: {
            beta_stress_sleep: "daily stress sleep longitudinal effect size",
          },
        });
      }
      throw new Error(`ENOENT: ${path}`);
    });

    streamTextMock.mockImplementation(({ messages }) => ({
      toUIMessageStreamResponse: () =>
        new Response(
          JSON.stringify({
            messages,
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
    }));

    const response = await POST(
      new Request("http://localhost/api/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "user-123",
          stageId: "stage-4",
          messages: [
            {
              id: "user-1",
              role: "user",
              parts: [{ type: "text", text: "Tighten the stress to sleep prior." }],
            },
          ],
        }),
      }),
    );

    expect(response.status).toBe(200);
    const payload = await response.json();
    expect(payload.messages).toHaveLength(5);
    expect(payload.messages[0]).toMatchObject({
      role: "system",
      content: "Comment on the Stage 6 results.",
    });
    expect(payload.messages[2]).toMatchObject({
      role: "system",
    });
    expect(String(payload.messages[2].content)).toContain("live refinement path");
    expect(payload.messages[3]).toMatchObject({
      role: "user",
    });
    expect(String(payload.messages[3].content)).toContain("All current Stage 4 decisions are shown together");
    expect(String(payload.messages[3].content)).toContain("## Your Decisions");
    expect(String(payload.messages[3].content)).toContain("beta_stress_sleep");
    expect(String(payload.messages[3].content)).not.toContain("## Full Current model_spec");
    expect(payload.messages[4]).toMatchObject({
      role: "user",
    });
  });

});
