import type { AnalysisManifest, AnalysisStageRun, AnalysisStageRuns } from "@/lib/api/analysis";
import {
  getEpisodeStatus,
  getEpisodeTimeline,
  type EpisodeStatus,
  type TransitionRecord,
} from "@/lib/server/episode-runs";
import { isSharedWorkspaceId } from "@/lib/shared-workspaces";
import { isStorageNotFoundError, prefixExists, readData } from "@/lib/storage";
import { STAGES, type StageId } from "@nof1-causal-lab/api-types";

function emptyStageRun(): AnalysisStageRun {
  return { execution: null };
}

function completedPersistedStageRun(createdAt: string): AnalysisStageRun {
  return {
    execution: {
      stateType: "COMPLETED",
      startTime: createdAt,
      endTime: createdAt,
    },
  };
}

function isStageId(value: unknown): value is StageId {
  return typeof value === "string" && STAGES.some((stage) => stage.id === value);
}

async function readWorkspaceQuestion(workspaceId: string): Promise<string | undefined> {
  try {
    const text = await readData(`${workspaceId}/query.txt`);
    const trimmed = text.trim();
    return trimmed || undefined;
  } catch (e: unknown) {
    if (isStorageNotFoundError(e)) {
      return undefined;
    }
    throw e;
  }
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

async function listPersistedStageIds(workspaceId: string): Promise<StageId[]> {
  const checks = await Promise.all(
    STAGES.map(async (stage) => ({
      exists: await prefixExists(`${workspaceId}/run/${stage.id}.json`),
      stageId: stage.id,
    })),
  );

  return checks.filter((entry) => entry.exists).map((entry) => entry.stageId);
}

/**
 * Manifest for curated shared workspaces (DEMO, GOLDEN, ...): pure persisted
 * artifacts, no episode journal.
 */
async function buildPersistedArtifactManifest(
  workspaceId: string,
): Promise<Omit<AnalysisManifest, "readOnly"> | null> {
  const [storedQuestion, persistedStageIds] = await Promise.all([
    readWorkspaceQuestion(workspaceId),
    listPersistedStageIds(workspaceId),
  ]);
  if (persistedStageIds.length === 0) return null;

  const completedStageIds = new Set<StageId>(persistedStageIds);
  const createdAt = new Date(0).toISOString();
  const stages = Object.fromEntries(
    STAGES.map((stage) => [
      stage.id,
      completedStageIds.has(stage.id) ? completedPersistedStageRun(createdAt) : emptyStageRun(),
    ]),
  ) as AnalysisStageRuns;

  return {
    workspaceId,
    createdAt,
    question: storedQuestion,
    stages,
  };
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

export async function buildAnalysisManifest(
  workspaceId: string,
): Promise<Omit<AnalysisManifest, "readOnly"> | null> {
  if (isSharedWorkspaceId(workspaceId)) {
    return buildPersistedArtifactManifest(workspaceId);
  }

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
