import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { STAGE_IDS } from "@nof1-causal-lab/api-types";
import { NextResponse } from "next/server";
import { isStorageNotFoundError, readData } from "@/lib/storage";
import { proposeMove } from "@/lib/server/episode-runs";
import { isRecord } from "@/lib/utils/type-guards";
import {
  mergePersistedTrace,
  summarizeRefinementMessages,
  type RefinementUIMessage,
} from "@/lib/utils/trace-to-core";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

/**
 * POST /api/refine/apply
 *
 * Materializes the client-held refinement payload. Earlier stages trigger
 * replay (write move + auto-run); the terminal stage writes the merged
 * result in place as a human-provenance artifact version.
 *
 * Body: { workspaceId, stageId, stagePatch?, messages? }
 */
export async function POST(request: Request) {
  const { workspaceId, stageId, stagePatch, messages } = await request.json();

  if (!workspaceId || !stageId) {
    return NextResponse.json({ error: "Missing workspaceId or stageId" }, { status: 400 });
  }

  const safeStageId = stageId.trim();
  if (!safeStageId || /[\\/]/.test(safeStageId)) {
    return NextResponse.json({ error: "Invalid stageId format" }, { status: 400 });
  }
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId, {
    requireMutable: true,
  });
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: safeWorkspaceId } = workspaceAccess;

  const isTerminalStage = safeStageId === STAGE_IDS.at(-1);
  const safeStagePatch = isRecord(stagePatch) ? stagePatch : {};
  const refinementMessages = Array.isArray(messages) ? (messages as RefinementUIMessage[]) : [];

  let currentStageData: Record<string, unknown>;
  try {
    currentStageData = JSON.parse(await readData(`${safeWorkspaceId}/run/${safeStageId}.json`));
  } catch (e: unknown) {
    if (isStorageNotFoundError(e)) {
      return NextResponse.json({ error: "Could not read current stage data" }, { status: 404 });
    }
    return NextResponse.json(
      { error: `Failed to read stage data: ${e instanceof Error ? e.message : String(e)}` },
      { status: 500 },
    );
  }

  const { llm_trace: existingTrace, _live: _liveField, ...originalDomain } = currentStageData;
  const refinementSummary = summarizeRefinementMessages(refinementMessages);
  const mergedStagePatch = {
    ...refinementSummary.stagePatch,
    ...safeStagePatch,
  };
  const baseTrace = isRecord(existingTrace) ? (existingTrace as unknown as LLMTrace) : null;
  const materializedPatch = {
    ...mergedStagePatch,
    ...(refinementMessages.length > 0
      ? {
          llm_trace: mergePersistedTrace(baseTrace, refinementMessages, {
            durationSeconds: refinementSummary.durationSeconds,
            usage: refinementSummary.usage,
          }),
        }
      : {}),
  };

  if (Object.keys(materializedPatch).length === 0) {
    return NextResponse.json({ error: "Nothing to materialize" }, { status: 400 });
  }

  if (isTerminalStage) {
    try {
      const outcome = await proposeMove(
        safeWorkspaceId,
        { kind: "write", artifact_id: "baseline_ranking", provenance: "human" },
        { ...originalDomain, llm_trace: existingTrace, ...materializedPatch },
      );
      if (outcome.status !== "applied") {
        return NextResponse.json(
          { error: `Persist failed: ${outcome.error_message ?? outcome.reason ?? outcome.status}` },
          { status: 502 },
        );
      }

      return NextResponse.json({
        ok: true,
        updatedFields: Object.keys(materializedPatch),
      });
    } catch (err) {
      return NextResponse.json(
        {
          error: `Persist failed: ${err instanceof Error ? err.message : String(err)}`,
        },
        { status: 500 },
      );
    }
  }

  const merged = { ...originalDomain, ...materializedPatch };

  try {
    // Trigger replay
    const replayHeaders = new Headers({
      "Content-Type": "application/json",
    });
    const cookie = request.headers.get("cookie");
    if (cookie) {
      replayHeaders.set("cookie", cookie);
    }

    const replayRes = await fetch(new URL("/api/replay", request.url), {
      method: "POST",
      headers: replayHeaders,
      body: JSON.stringify({
        workspaceId: safeWorkspaceId,
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
      updatedFields: Object.keys(materializedPatch),
      ...replayResult,
    });
  } catch (err) {
    return NextResponse.json(
      {
        error: `Apply failed: ${err instanceof Error ? err.message : String(err)}`,
      },
      { status: 500 },
    );
  }
}
