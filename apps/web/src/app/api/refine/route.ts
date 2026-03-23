import { INTERACTIVE_STAGES, STAGE_TOOLS } from "@causal-ssm/api-types";
import type { LLMTrace } from "@causal-ssm/api-types";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { jsonSchema, streamText, tool } from "ai";
import { basename } from "node:path";
import { NextResponse } from "next/server";

import { resolveApiKey } from "@/lib/api/resolve-api-key";
import { getToolServerUrl } from "@/lib/runtime-urls";
import { readData, writeData, ensureDir } from "@/lib/storage";
import { traceToModelMessages } from "@/lib/utils/trace-to-core";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

const TOOL_SERVER = getToolServerUrl();

/**
 * POST /api/refine
 *
 * Streams a refinement conversation with full pipeline trace context
 * and the same tools the pipeline used (proxied to Python for execution).
 *
 * Body: { messages, workspaceId, stageId }
 */
export async function POST(req: Request) {
  const resolved = resolveApiKey(req);
  if (resolved.error) {
    return NextResponse.json({ error: resolved.error }, { status: resolved.status });
  }

  const { messages, workspaceId, stageId } = await req.json();
  const safeStageId = typeof stageId === "string" ? basename(stageId.trim()) : "";

  let normalizedWorkspaceId = "";
  if (workspaceId) {
    const workspaceAccess = await requireWorkspaceAccess(req, workspaceId);
    if (!workspaceAccess.ok) {
      return workspaceAccess.response;
    }
    normalizedWorkspaceId = workspaceAccess.workspaceId;
  }

  if (stageId && (!safeStageId || safeStageId !== stageId.trim())) {
    return NextResponse.json({ error: "Invalid stageId format" }, { status: 400 });
  }

  let traceContext: ReturnType<typeof traceToModelMessages> = [];
  if (normalizedWorkspaceId && safeStageId) {
    try {
      const raw = await readData(`${normalizedWorkspaceId}/run/${safeStageId}.json`);
      const stageData = JSON.parse(raw);

      if (stageData.llm_trace) {
        const trace: LLMTrace = stageData.llm_trace;
        traceContext = traceToModelMessages(trace.messages);
      }
    } catch {
      // No trace available — proceed without context
    }
  }

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
          const res = await fetch(
            `${TOOL_SERVER}/api/tools/${safeStageId}/${t.name}`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ workspace_id: normalizedWorkspaceId, input: args }),
            },
          );
          if (!res.ok) {
            const text = await res.text();
            throw new Error(`Tool execution failed: ${text}`);
          }
          const data = await res.json();

          if (data.stage_output) {
            await ensureDir(`${normalizedWorkspaceId}/run`);
            await writeData(
              `${normalizedWorkspaceId}/run/${safeStageId}-draft.json`,
              JSON.stringify(data.stage_output),
            );
          }

          return data.result;
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any),
    ]),
  );

  const openrouter = createOpenRouter({ apiKey: resolved.key });

  const result = streamText({
    model: openrouter("anthropic/claude-sonnet-4"),
    messages: [...traceContext, ...messages],
    ...(Object.keys(tools).length > 0 ? { tools, maxSteps: 10 } : {}),
  });

  return result.toUIMessageStreamResponse();
}
