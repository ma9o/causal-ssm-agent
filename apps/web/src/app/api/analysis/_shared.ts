import type { AnalysisManifest, AnalysisStageRun, AnalysisStageRuns } from "@/lib/api/analysis";
import {
  getEpisodeStatus,
  getEpisodeTimeline,
  type EpisodeStatus,
  type TransitionRecord,
} from "@/lib/server/episode-runs";
import { isStorageNotFoundError, readData } from "@/lib/storage";
import { STAGES, type StageId } from "@nof1-causal-lab/api-types";

function emptyStageRun(): AnalysisStageRun {
  return { execution: null };
}

function isStageId(value: unknown): value is StageId {
  return typeof value === "string" && STAGES.some((stage) => stage.id === value);
}

async function readEpisodeQuestion(
  workspaceId: string,
  status: EpisodeStatus,
): Promise<string | undefined> {
  const version = status.state.current.question?.version;
  if (version == null) {
    return undefined;
  }

  try {
    const raw = await readData(`${workspaceId}/store/question/v${version}/question.json`);
    const parsed = JSON.parse(raw) as { text?: unknown };
    return typeof parsed.text === "string" && parsed.text.trim() ? parsed.text.trim() : undefined;
  } catch (e: unknown) {
    if (isStorageNotFoundError(e)) {
      return undefined;
    }
    throw e;
  }
}

/**
 * Per-stage execution summaries from the episode journal: the latest
 * completed run attempt per stage wins (applied → completed, raised →
 * failed; rejected attempts never executed).
 */
function summarizeTimelineStageRuns(transitions: TransitionRecord[]): AnalysisStageRuns {
  const stages = Object.fromEntries(
    STAGES.map((stage) => [stage.id, emptyStageRun()]),
  ) as AnalysisStageRuns;

  for (const record of transitions) {
    if (record.move.kind !== "run" || !isStageId(record.move.stage_id)) {
      continue;
    }
    if (record.status === "rejected") {
      continue;
    }

    stages[record.move.stage_id] = {
      execution: {
        stateType: record.status === "applied" ? "COMPLETED" : "FAILED",
        startTime: record.ts,
        endTime: record.ts,
      },
    };
  }

  return stages;
}

/**
 * The manifest comes straight from the episode journal — a published
 * (read-only) workspace carries its journal along, so there is no separate
 * curated-demo path.
 */
export async function buildAnalysisManifest(
  workspaceId: string,
): Promise<Omit<AnalysisManifest, "readOnly"> | null> {
  const [status, timeline] = await Promise.all([
    getEpisodeStatus(workspaceId),
    getEpisodeTimeline(workspaceId),
  ]);
  if (timeline.transitions.length === 0) {
    return null;
  }

  const question = await readEpisodeQuestion(workspaceId, status);

  return {
    workspaceId,
    createdAt: timeline.transitions[0].ts,
    question,
    stages: summarizeTimelineStageRuns(timeline.transitions),
  };
}
