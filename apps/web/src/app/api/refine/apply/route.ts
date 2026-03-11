import type { LLMTrace } from "@causal-ssm/api-types";
import { openrouter } from "@openrouter/ai-sdk-provider";
import { generateText, jsonSchema, tool } from "ai";
import { readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { NextResponse } from "next/server";

import { traceToCoreMessages } from "@/lib/utils/trace-to-core";

const RESULTS_DIR = process.cwd() + "/../data-pipeline/results";

/**
 * POST /api/refine/apply
 *
 * Takes the refinement conversation and asks the LLM to produce
 * the updated stage data using structured tool_use extraction.
 * Then sends it to /api/replay to overwrite and trigger downstream re-execution.
 *
 * Body: { messages, runId, stageId }
 */
export async function POST(request: Request) {
  const { messages, runId, stageId } = await request.json();

  if (!messages || !runId || !stageId) {
    return NextResponse.json(
      { error: "Missing messages, runId, or stageId" },
      { status: 400 },
    );
  }

  const safeRunId = basename(runId);
  const safeStageId = basename(stageId);
  const stagePath = resolve(join(RESULTS_DIR, safeRunId, `${safeStageId}.json`));

  let currentData: Record<string, unknown>;
  try {
    const raw = await readFile(stagePath, "utf-8");
    currentData = JSON.parse(raw);
  } catch {
    return NextResponse.json({ error: "Could not read current stage data" }, { status: 404 });
  }

  // Build trace context for the extraction LLM
  let traceContext: ReturnType<typeof traceToCoreMessages> = [];
  if (currentData.llm_trace) {
    const trace = currentData.llm_trace as LLMTrace;
    traceContext = traceToCoreMessages(trace.messages);
  }

  // Build a JSON Schema for the update_stage tool from the current data's structure.
  // This guides the LLM to produce output matching the stage contract shape.
  const { llm_trace: _trace, outcome: _outcome, ...domainFields } = currentData;
  const properties: Record<string, unknown> = {};
  for (const key of Object.keys(domainFields)) {
    properties[key] = {}; // any type — the Python contract validates on the other end
  }

  const updateSchema = {
    type: "object" as const,
    properties: {
      stage_data: {
        type: "object" as const,
        description: `Complete updated stage output with keys: ${Object.keys(domainFields).join(", ")}`,
        properties,
        required: Object.keys(domainFields),
      },
    },
    required: ["stage_data"],
  };

  try {
    const result = await generateText({
      model: openrouter("anthropic/claude-sonnet-4"),
      system: [
        `You are updating stage "${safeStageId}" output based on a refinement conversation.`,
        `Current data:\n${JSON.stringify(domainFields, null, 2)}`,
        "Call update_stage with the COMPLETE updated output, preserving all fields.",
      ].join("\n\n"),
      messages: [
        ...traceContext,
        ...messages,
        {
          role: "user" as const,
          content: "Call update_stage with the complete updated stage output incorporating all changes discussed.",
        },
      ],
      tools: {
        update_stage: tool({
          description: "Produce the complete updated stage output",
          parameters: jsonSchema(updateSchema),
        }),
      },
      maxSteps: 1,
    });

    // Extract from tool call
    const toolCall = result.toolCalls?.[0];
    if (!toolCall) {
      return NextResponse.json(
        { error: "LLM did not call update_stage tool" },
        { status: 422 },
      );
    }

    const updatedData = (toolCall.args as { stage_data: Record<string, unknown> }).stage_data;

    // Merge: keep fields the LLM didn't touch, update the ones it did
    const merged = { ...domainFields, ...updatedData };

    // Call the replay endpoint
    const replayRes = await fetch(new URL("/api/replay", request.url), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        runId: safeRunId,
        stageId: safeStageId,
        stageData: merged,
      }),
    });

    if (!replayRes.ok) {
      const error = await replayRes.text();
      return NextResponse.json({ error: `Replay failed: ${error}` }, { status: 502 });
    }

    const replayResult = await replayRes.json();
    return NextResponse.json({
      ok: true,
      updatedFields: Object.keys(updatedData),
      ...replayResult,
    });
  } catch (err) {
    return NextResponse.json(
      { error: `Apply failed: ${err instanceof Error ? err.message : String(err)}` },
      { status: 500 },
    );
  }
}
