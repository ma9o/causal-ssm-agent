import { openrouter } from "@openrouter/ai-sdk-provider";
import { generateText } from "ai";
import { NextResponse } from "next/server";

const RESULTS_DIR = process.cwd() + "/../data-pipeline/results";

/**
 * POST /api/refine/apply
 *
 * Takes the refinement conversation and asks the LLM to produce
 * the updated stage data as structured JSON. Then sends it to
 * /api/replay to overwrite and trigger downstream re-execution.
 */
export async function POST(request: Request) {
  const { messages, runId, stageId } = await request.json();

  if (!messages || !runId || !stageId) {
    return NextResponse.json(
      { error: "Missing messages, runId, or stageId" },
      { status: 400 },
    );
  }

  // Read the current stage data to understand the schema
  const { readFile } = await import("node:fs/promises");
  const { basename, join, resolve } = await import("node:path");

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

  // Ask the LLM to produce the updated stage data
  const extractionMessages = [
    ...messages,
    {
      role: "user" as const,
      content: `Based on our conversation above, produce the COMPLETE updated stage output as a single JSON object. The current output has these top-level keys: ${Object.keys(currentData).join(", ")}. Return the full object with all fields, incorporating all changes we discussed. Only include the JSON, no explanation.`,
    },
  ];

  try {
    const { text } = await generateText({
      model: openrouter("anthropic/claude-sonnet-4"),
      messages: extractionMessages,
    });

    // Extract JSON from the response (may be wrapped in ```json blocks)
    const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) ?? [null, text];
    const jsonStr = (jsonMatch[1] ?? text).trim();
    let updatedData: Record<string, unknown>;
    try {
      updatedData = JSON.parse(jsonStr);
    } catch {
      return NextResponse.json(
        { error: "Failed to parse LLM output as JSON", raw: text },
        { status: 422 },
      );
    }

    // Merge: keep fields the LLM didn't touch, update the ones it did
    // Remove internal fields that shouldn't be overwritten
    const { llm_trace, outcome, ...existingFields } = currentData;
    const merged = { ...existingFields, ...updatedData };

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
      updatedFields: Object.keys(updatedData as Record<string, unknown>),
      ...replayResult,
    });
  } catch (err) {
    return NextResponse.json(
      { error: `Apply failed: ${err instanceof Error ? err.message : String(err)}` },
      { status: 500 },
    );
  }
}
