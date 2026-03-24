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

vi.mock("@/lib/api/resolve-api-key", () => ({
  resolveApiKey: vi.fn(() => ({ key: "test-key" })),
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
  tool: vi.fn((definition) => definition),
  streamText: streamTextMock,
}));

import { readData } from "@/lib/storage";
import {
  refinementNeedsActivation,
  refinementRequiresConfirmation,
} from "@/lib/contexts/refinement-context";
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

  it("keeps terminal stage activation separate from downstream invalidation confirmation", () => {
    expect(refinementNeedsActivation("stage-6", null)).toBe(true);
    expect(refinementRequiresConfirmation("stage-6", null)).toBe(false);
    expect(refinementNeedsActivation("stage-6", "stage-6")).toBe(false);

    expect(refinementRequiresConfirmation("stage-1a", null)).toBe(true);
    expect(refinementRequiresConfirmation("stage-1a", "stage-6")).toBe(true);
  });
});
