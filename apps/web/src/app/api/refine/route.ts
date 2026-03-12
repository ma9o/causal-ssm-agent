import { INTERACTIVE_STAGES, STAGE_TOOLS } from "@causal-ssm/api-types";
import type { LLMTrace } from "@causal-ssm/api-types";
import { openrouter } from "@openrouter/ai-sdk-provider";
import { jsonSchema, streamText, tool } from "ai";
import { basename, join, resolve } from "node:path";
import { readFile, writeFile, mkdir } from "node:fs/promises";

import { traceToCoreMessages } from "@/lib/utils/trace-to-core";

const RESULTS_DIR = process.cwd() + "/../data-pipeline/results";
const TOOL_SERVER = process.env.TOOL_SERVER_URL ?? "http://localhost:8100";

/**
 * POST /api/refine
 *
 * Streams a refinement conversation with full pipeline trace context
 * and the same tools the pipeline used (proxied to Python for execution).
 *
 * Body: { messages, runId, stageId }
 */
export async function POST(req: Request) {
  const { messages, runId, stageId } = await req.json();

  // Build trace context if we have a run and stage
  let traceContext: ReturnType<typeof traceToCoreMessages> = [];
  if (runId && stageId) {
    try {
      const safeRunId = basename(runId);
      const safeStageId = basename(stageId);
      const stagePath = resolve(
        join(RESULTS_DIR, safeRunId, `${safeStageId}.json`),
      );
      const raw = await readFile(stagePath, "utf-8");
      const stageData = JSON.parse(raw);

      if (stageData.llm_trace) {
        const trace: LLMTrace = stageData.llm_trace;
        traceContext = traceToCoreMessages(trace.messages);
      }
    } catch {
      // No trace available — proceed without context
    }
  }

  // Build tools if this is an interactive stage
  const toolDefs =
    stageId && INTERACTIVE_STAGES.includes(stageId)
      ? STAGE_TOOLS[stageId] ?? []
      : [];

  const tools = Object.fromEntries(
    toolDefs.map((t) => [
      t.name,
      tool({
        description: t.description,
        parameters: jsonSchema(t.parameters),
        execute: async (args) => {
          const safeRunId = basename(runId);
          const safeStageId = basename(stageId);
          const res = await fetch(
            `${TOOL_SERVER}/api/tools/${stageId}/${t.name}`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ run_id: runId, input: args }),
            },
          );
          if (!res.ok) {
            const text = await res.text();
            throw new Error(`Tool execution failed: ${text}`);
          }
          const data = await res.json();

          // Persist draft on successful tool call (stage_output is set)
          if (data.stage_output) {
            const draftDir = resolve(join(RESULTS_DIR, safeRunId));
            await mkdir(draftDir, { recursive: true });
            await writeFile(
              resolve(join(draftDir, `${safeStageId}-draft.json`)),
              JSON.stringify(data.stage_output),
            );
          }

          return data.result;
        },
      }),
    ]),
  );

  const result = streamText({
    model: openrouter("anthropic/claude-sonnet-4"),
    messages: [...traceContext, ...messages],
    ...(Object.keys(tools).length > 0 ? { tools, maxSteps: 10 } : {}),
  });

  return result.toUIMessageStreamResponse();
}
