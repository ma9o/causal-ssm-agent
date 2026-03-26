import { INTERACTIVE_STAGES, STAGE_TOOLS } from "@causal-ssm/api-types";
import type { LLMTrace } from "@causal-ssm/api-types";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { convertToModelMessages, jsonSchema, stepCountIs, streamText, tool } from "ai";
import { NextResponse } from "next/server";

import { getToolServerUrl } from "@/lib/runtime-urls";
import { resolveOpenRouterAccess } from "@/lib/server/openrouter-access";
import { readData } from "@/lib/storage";
import {
  type RefinementMessageMetadata,
  type RefinementUIMessage,
  traceToModelMessages,
} from "@/lib/utils/trace-to-core";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

const TOOL_SERVER = getToolServerUrl();
const REFINE_MODEL = "anthropic/claude-sonnet-4";

async function loadTraceContext(
  workspaceId: string,
  stageId: string,
): Promise<{
  traceContext: ReturnType<typeof traceToModelMessages>;
}> {
  try {
    const raw = await readData(`${workspaceId}/run/${stageId}.json`);
    const stageData = JSON.parse(raw);

    if (stageData.llm_trace) {
      const baseTrace = stageData.llm_trace as LLMTrace;
      return {
        traceContext: traceToModelMessages(baseTrace.messages),
      };
    }
  } catch {
    // No trace available — proceed without context
  }

  return {
    traceContext: [],
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeToolArgsForSchema(
  args: Record<string, unknown>,
  schema: unknown,
): Record<string, unknown> {
  if (!isRecord(schema) || !isRecord(schema.properties)) {
    return args;
  }

  const normalized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    const propertySchema = schema.properties[key];
    if (isRecord(propertySchema) && propertySchema.type === "string" && typeof value !== "string") {
      normalized[key] = JSON.stringify(value);
      continue;
    }

    normalized[key] = value;
  }

  return normalized;
}

/**
 * POST /api/refine
 *
 * Streams a refinement conversation with full pipeline trace context
 * and the same tools the pipeline used (proxied to Python for execution).
 *
 * Body: { messages, workspaceId, stageId }
 */
export async function POST(req: Request) {
  const access = await resolveOpenRouterAccess();
  if (access.mode === "none") {
    const error =
      access.reason === "trial_exhausted"
        ? "Trial credits exhausted. Sign in with OpenRouter to continue."
        : "No OpenRouter access is configured.";
    return NextResponse.json({ error }, { status: 402 });
  }

  const { messages, workspaceId, stageId, pendingStagePatch } = await req.json();
  if (!Array.isArray(messages)) {
    return NextResponse.json({ error: "messages must be an array" }, { status: 400 });
  }

  const uiMessages = messages as RefinementUIMessage[];
  const hasWorkspaceId = typeof workspaceId === "string" && workspaceId.trim().length > 0;
  const safeStageId = typeof stageId === "string" ? stageId.trim() : "";
  const safePendingStagePatch = isRecord(pendingStagePatch) ? pendingStagePatch : {};

  if (stageId && (!safeStageId || /[\\/]/.test(safeStageId))) {
    return NextResponse.json({ error: "Invalid stageId format" }, { status: 400 });
  }
  const workspaceAccess = hasWorkspaceId ? await requireWorkspaceAccess(req, workspaceId) : null;
  if (workspaceAccess && !workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const normalizedWorkspaceId = workspaceAccess?.ok ? workspaceAccess.workspaceId : null;

  const { traceContext } =
    normalizedWorkspaceId && safeStageId
      ? await loadTraceContext(normalizedWorkspaceId, safeStageId)
      : { traceContext: [] };

  // Build tools if this is an interactive stage
  const toolDefs =
    normalizedWorkspaceId && safeStageId && INTERACTIVE_STAGES.includes(safeStageId)
      ? STAGE_TOOLS[safeStageId] ?? []
      : [];

  const tools = Object.fromEntries(
    toolDefs.map((t) => [
      t.name,
      tool({
        description: t.description,
        parameters: jsonSchema(t.parameters),
        execute: async (args: Record<string, unknown>) => {
          if (!normalizedWorkspaceId) {
            throw new Error("Tool execution requires a workspace");
          }
          const normalizedArgs = normalizeToolArgsForSchema(args, t.parameters);
          const res = await fetch(
            `${TOOL_SERVER}/api/tools/${safeStageId}/${t.name}`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                workspace_id: normalizedWorkspaceId,
                input: normalizedArgs,
              }),
            },
          );
          if (!res.ok) {
            const text = await res.text();
            throw new Error(`Tool execution failed: ${text}`);
          }
          const data = await res.json();

          if (data.stage_output) {
            nextStagePatch = {
              ...nextStagePatch,
              ...data.stage_output,
            };
          }

          return data.result;
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any),
    ]),
  );

  const modelMessages = await convertToModelMessages(uiMessages);
  const startedAt = Date.now();
  let nextStagePatch = { ...safePendingStagePatch };
  const openrouter = createOpenRouter({ apiKey: access.apiKey });

  const result = streamText({
    model: openrouter(REFINE_MODEL),
    messages: [...traceContext, ...modelMessages],
    ...(Object.keys(tools).length > 0 ? { tools, stopWhen: stepCountIs(10) } : {}),
  });

  return result.toUIMessageStreamResponse({
    originalMessages: uiMessages,
    messageMetadata: ({ part }): RefinementMessageMetadata | undefined => {
      if (part.type !== "finish") {
        return undefined;
      }

      return {
        durationSeconds: (Date.now() - startedAt) / 1000,
        stagePatch: nextStagePatch,
        usage: {
          inputTokens: part.totalUsage.inputTokens ?? undefined,
          outputTokens: part.totalUsage.outputTokens ?? undefined,
          reasoningTokens:
            part.totalUsage.outputTokenDetails.reasoningTokens ??
            part.totalUsage.reasoningTokens ??
            undefined,
        },
      };
    },
  });
}
