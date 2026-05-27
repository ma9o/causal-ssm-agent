import type { AnalysisManifest, AnalysisStageRun, AnalysisStageRuns } from "@/lib/api/analysis";
import {
  STAGE2_EVENT_PREFIX,
  reduceStage2Events,
  type PrefectStage2EventRecord,
  type Stage2ReplayState,
} from "@/lib/stage2-runtime";
import {
  STAGE4_EVENT_PREFIX,
  reduceStage4Events,
  type PrefectStage4EventRecord,
  type Stage4ReplayState,
} from "@/lib/stage4-runtime";
import { dedupeRootFlowRunIds, getWorkspaceRunTag } from "@/lib/root-flow-runs";
import {
  type StageExecutionSummary,
  STAGE_PROGRESS_EVENT_FILTER_PREFIX,
  type StageProgressStatus,
  summarizeStageProgressEvents,
} from "@/lib/stage-runtime";
import { getStageLogScopePolicy } from "@/lib/stage-observability";
import { prefectFetch } from "@/lib/server/prefect-runs";
import { isSharedWorkspaceId } from "@/lib/shared-workspaces";
import { isStorageNotFoundError, prefixExists, readData } from "@/lib/storage";
import { STAGES, type StageId } from "@nof1-causal-lab/api-types";
import { getPrefectApiUrl } from "@/lib/runtime-urls";

const PREFECT_API = getPrefectApiUrl();

interface PrefectFlowRun {
  id: string;
  parameters?: Record<string, unknown>;
  created?: string | null;
  start_time?: string | null;
  expected_start_time?: string | null;
}

interface PrefectEventPage {
  events?: PrefectEvent[];
  next_page?: string | null;
}

interface PrefectEvent {
  event?: string | null;
  occurred?: string | null;
  payload?: Record<string, unknown>;
}

interface RootFlowRunLineageEntry {
  rootFlowRunId: string;
  startStage: StageId;
  endStage: StageId;
  createdAt: string | null;
  query: string | null;
}

interface PersistedSession {
  createdAt?: string;
  rootFlowRunIds: string[];
}

function emptyStageRun(): AnalysisStageRun {
  return {
    ownerRootFlowRunId: null,
    stageSubflowRunId: null,
    initialLogFlowRunIds: [],
    execution: null,
  };
}

function completedPersistedStageRun(createdAt: string): AnalysisStageRun {
  return {
    ownerRootFlowRunId: null,
    stageSubflowRunId: null,
    initialLogFlowRunIds: [],
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

function getFlowRunStartStage(parameters?: Record<string, unknown>): StageId {
  return isStageId(parameters?.start_stage) ? parameters.start_stage : "stage-0";
}

function getFlowRunEndStage(parameters?: Record<string, unknown>): StageId {
  return isStageId(parameters?.end_stage) ? parameters.end_stage : STAGES[STAGES.length - 1].id;
}

function getStageIndex(stageId: StageId): number {
  return STAGES.findIndex((stage) => stage.id === stageId);
}

function getStageOwningRootFlowRunId(
  lineage: RootFlowRunLineageEntry[],
  stageExecutionsByRootFlowRunId: Map<string, Partial<Record<StageId, StageExecutionSummary>>>,
  stageId: StageId,
): string | null {
  const stageIndex = getStageIndex(stageId);
  if (stageIndex === -1) return null;

  let ownerRootFlowRunId: string | null = null;

  for (const entry of lineage) {
    const startIndex = getStageIndex(entry.startStage);
    const endIndex = getStageIndex(entry.endStage);
    const stageExecutions = stageExecutionsByRootFlowRunId.get(entry.rootFlowRunId);
    const hasAnyStageExecution = !!stageExecutions && Object.keys(stageExecutions).length > 0;
    if (hasAnyStageExecution && startIndex <= stageIndex && stageIndex <= endIndex) {
      ownerRootFlowRunId = entry.rootFlowRunId;
    }
  }

  return ownerRootFlowRunId;
}

function isStageProgressStatus(value: unknown): value is StageProgressStatus {
  return value === "running" || value === "completed" || value === "failed";
}

function summarizeStageExecutions(
  events: PrefectEvent[],
): Partial<Record<StageId, StageExecutionSummary>> {
  const byStageId = new Map<StageId, PrefectEvent[]>();

  for (const event of events) {
    const stageId = event.payload?.stage_id;
    const status = event.payload?.status;
    if (!isStageId(stageId) || !isStageProgressStatus(status)) {
      continue;
    }
    const stageEvents = byStageId.get(stageId) ?? [];
    stageEvents.push(event);
    byStageId.set(stageId, stageEvents);
  }

  return Object.fromEntries(
    [...byStageId.entries()]
      .map(
        ([stageId, stageEvents]) =>
          [
            stageId,
            summarizeStageProgressEvents(
              stageEvents.map((event) => ({
                status: event.payload?.status as StageProgressStatus,
                occurred: event.occurred,
                stageSubflowRunId: event.payload?.stage_subflow_run_id,
                logFlowRunIds: event.payload?.log_flow_run_ids,
              })),
            ),
          ] as const,
      )
      .filter((entry): entry is readonly [StageId, StageExecutionSummary] => entry[1] !== null),
  ) as Partial<Record<StageId, StageExecutionSummary>>;
}

type PrefectResult<T> =
  | { ok: true; data: T }
  | { ok: false; status: number; statusText: string };

async function prefectGetJson<T>(path: string): Promise<PrefectResult<T>> {
  const response = await prefectFetch(`${PREFECT_API}${path}`, { cache: "no-store" });
  if (!response.ok) return { ok: false, status: response.status, statusText: response.statusText };
  return { ok: true, data: await response.json() };
}

async function prefectPostJson<T>(path: string, body: unknown): Promise<PrefectResult<T>> {
  const response = await prefectFetch(`${PREFECT_API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) return { ok: false, status: response.status, statusText: response.statusText };
  return { ok: true, data: await response.json() };
}

async function fetchWorkspaceRootFlowRunIds(workspaceId: string): Promise<string[]> {
  const result = await prefectPostJson<PrefectFlowRun[]>("/flow_runs/filter", {
    flow_runs: {
      tags: { all_: [getWorkspaceRunTag(workspaceId)] },
      parent_task_run_id: { is_null_: true },
    },
    sort: "START_TIME_DESC",
    limit: 200,
  });
  const flowRuns = result.ok ? result.data : [];

  return flowRuns.map((flowRun) => flowRun.id);
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

function parsePersistedRootFlowRunIds(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
  );
}

async function readPersistedSession(workspaceId: string): Promise<PersistedSession | null> {
  try {
    const raw = await readData(`${workspaceId}/session.json`);
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const createdAt = typeof parsed.createdAt === "string" ? parsed.createdAt : undefined;
    return {
      createdAt,
      rootFlowRunIds: parsePersistedRootFlowRunIds(parsed.rootFlowRunIds),
    };
  } catch (e: unknown) {
    if (isStorageNotFoundError(e)) {
      return null;
    }
    return null;
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

async function buildPersistedArtifactManifest(
  workspaceId: string,
  bootstrapRootFlowRunIds: string[] = [],
): Promise<AnalysisManifest | null> {
  const [storedQuestion, session, persistedStageIds] = await Promise.all([
    readWorkspaceQuestion(workspaceId),
    readPersistedSession(workspaceId),
    listPersistedStageIds(workspaceId),
  ]);
  if (persistedStageIds.length === 0) return null;

  const completedStageIds = new Set<StageId>(persistedStageIds);
  const createdAt = session?.createdAt ?? new Date(0).toISOString();
  const rootFlowRunIds = dedupeRootFlowRunIds([
    ...(session?.rootFlowRunIds ?? []),
    ...bootstrapRootFlowRunIds,
  ]);
  const latestRootFlowRunId = rootFlowRunIds.at(-1) ?? null;
  const stages = Object.fromEntries(
    STAGES.map((stage) => [
      stage.id,
      completedStageIds.has(stage.id)
        ? completedPersistedStageRun(createdAt)
        : emptyStageRun(),
    ]),
  ) as AnalysisStageRuns;

  return {
    workspaceId,
    createdAt,
    question: storedQuestion,
    rootFlowRunIds,
    latestRootFlowRunId,
    stages,
  };
}

async function mergePersistedStageRuns(
  workspaceId: string,
  stageRuns: AnalysisStageRuns,
  createdAt: string,
): Promise<AnalysisStageRuns> {
  const persistedStageIds = await listPersistedStageIds(workspaceId);
  if (persistedStageIds.length === 0) {
    return stageRuns;
  }

  const completedStageIds = new Set<StageId>(persistedStageIds);
  return Object.fromEntries(
    STAGES.map((stage) => {
      const existing = stageRuns[stage.id];
      return [
        stage.id,
        existing.execution || !completedStageIds.has(stage.id)
          ? existing
          : completedPersistedStageRun(createdAt),
      ];
    }),
  ) as AnalysisStageRuns;
}

async function fetchRootFlowRunLineage(
  rootFlowRunIds: string[],
): Promise<RootFlowRunLineageEntry[]> {
  return Promise.all(
    rootFlowRunIds.map(async (rootFlowRunId) => {
      const result = await prefectGetJson<PrefectFlowRun>(`/flow_runs/${rootFlowRunId}`);
      const flowRun = result.ok ? result.data : null;
      return {
        rootFlowRunId,
        startStage: getFlowRunStartStage(flowRun?.parameters),
        endStage: getFlowRunEndStage(flowRun?.parameters),
        createdAt: flowRun?.start_time ?? flowRun?.expected_start_time ?? flowRun?.created ?? null,
        query: typeof flowRun?.parameters?.query === "string" ? flowRun.parameters.query : null,
      };
    }),
  );
}

function getLineageTimestampMs(entry: RootFlowRunLineageEntry): number {
  if (!entry.createdAt) {
    return Number.NaN;
  }

  return Date.parse(entry.createdAt);
}

function sortLineageEntries(
  lineage: RootFlowRunLineageEntry[],
  inputOrder: readonly string[],
): RootFlowRunLineageEntry[] {
  const order = new Map(inputOrder.map((rootFlowRunId, index) => [rootFlowRunId, index]));

  return [...lineage].sort((left, right) => {
    const leftTs = getLineageTimestampMs(left);
    const rightTs = getLineageTimestampMs(right);

    if (Number.isFinite(leftTs) && Number.isFinite(rightTs) && leftTs !== rightTs) {
      return leftTs - rightTs;
    }
    if (Number.isFinite(leftTs) !== Number.isFinite(rightTs)) {
      return Number.isFinite(leftTs) ? -1 : 1;
    }

    return (order.get(left.rootFlowRunId) ?? 0) - (order.get(right.rootFlowRunId) ?? 0);
  });
}

async function fetchStageProgressEventsForRootFlowRun(
  rootFlowRunId: string,
): Promise<PrefectEvent[]> {
  return fetchPrefectEventsForRootFlowRun(rootFlowRunId, STAGE_PROGRESS_EVENT_FILTER_PREFIX, 50);
}

async function fetchPrefectEventsForRootFlowRun(
  rootFlowRunId: string,
  eventPrefix: string,
  limit = 5000,
): Promise<PrefectEvent[]> {
  const result = await prefectPostJson<PrefectEventPage>("/events/filter", {
    filter: {
      event: { prefix: [eventPrefix] },
      resource: { id: [`prefect.flow-run.${rootFlowRunId}`] },
      order: "ASC",
    },
    limit,
  });

  return (result.ok ? result.data.events : undefined) ?? [];
}

export async function buildStage4ReplayState(rootFlowRunId: string): Promise<Stage4ReplayState> {
  const events = await fetchPrefectEventsForRootFlowRun(rootFlowRunId, STAGE4_EVENT_PREFIX);
  return reduceStage4Events(events as PrefectStage4EventRecord[]);
}

export async function buildStage2ReplayState(rootFlowRunId: string): Promise<Stage2ReplayState> {
  const events = await fetchPrefectEventsForRootFlowRun(rootFlowRunId, STAGE2_EVENT_PREFIX, 10000);
  return reduceStage2Events(events as PrefectStage2EventRecord[]);
}

export async function resolveStageLogScopeFlowRunIds(
  stageId: StageId,
  stageSubflowRunId: string | null,
): Promise<string[]> {
  if (!stageSubflowRunId) {
    return [];
  }

  if (getStageLogScopePolicy(stageId) === "subflow") {
    return [stageSubflowRunId];
  }

  const childResult = await prefectPostJson<PrefectFlowRun[]>("/flow_runs/filter", {
    flow_runs: { parent_flow_run_id: { any_: [stageSubflowRunId] } },
    sort: "START_TIME_ASC",
    limit: 50,
  });
  const childFlowRuns = childResult.ok ? childResult.data : [];

  return [...new Set([stageSubflowRunId, ...childFlowRuns.map((flowRun) => flowRun.id)])];
}

async function buildStageRuns(
  rootFlowRunIds: string[],
  lineage?: RootFlowRunLineageEntry[],
): Promise<AnalysisStageRuns> {
  const effectiveLineage = lineage ?? (await fetchRootFlowRunLineage(rootFlowRunIds));
  const stageExecutionEntries = await Promise.all(
    effectiveLineage.map(
      async ({ rootFlowRunId }) =>
        [
          rootFlowRunId,
          summarizeStageExecutions(await fetchStageProgressEventsForRootFlowRun(rootFlowRunId)),
        ] as const,
    ),
  );
  const stageExecutionsByRootFlowRunId = new Map(stageExecutionEntries);

  const stageRuns = await Promise.all(
    STAGES.map(async (stage) => {
      const ownerRootFlowRunId = getStageOwningRootFlowRunId(
        effectiveLineage,
        stageExecutionsByRootFlowRunId,
        stage.id,
      );
      if (!ownerRootFlowRunId) {
        return [stage.id, emptyStageRun()] as const;
      }

      const stageExecution =
        stageExecutionsByRootFlowRunId.get(ownerRootFlowRunId)?.[stage.id] ?? null;

      return [
        stage.id,
        {
          ownerRootFlowRunId,
          stageSubflowRunId: stageExecution?.stageSubflowRunId ?? null,
          initialLogFlowRunIds: stageExecution?.initialLogFlowRunIds ?? [],
          execution: stageExecution?.execution ?? null,
        },
      ] as const;
    }),
  );

  return Object.fromEntries(stageRuns) as AnalysisStageRuns;
}

export async function buildAnalysisManifest(
  workspaceId: string,
  bootstrapRootFlowRunIds: string[] = [],
): Promise<AnalysisManifest | null> {
  if (isSharedWorkspaceId(workspaceId)) {
    return buildPersistedArtifactManifest(workspaceId, bootstrapRootFlowRunIds);
  }

  let prefectRootFlowRunIds: string[];
  try {
    prefectRootFlowRunIds = await fetchWorkspaceRootFlowRunIds(workspaceId);
  } catch {
    return buildPersistedArtifactManifest(workspaceId, bootstrapRootFlowRunIds);
  }
  const candidateRootFlowRunIds = dedupeRootFlowRunIds([
    ...prefectRootFlowRunIds,
    ...bootstrapRootFlowRunIds,
  ]);
  if (candidateRootFlowRunIds.length === 0) {
    return buildPersistedArtifactManifest(workspaceId, bootstrapRootFlowRunIds);
  }

  const [storedQuestion, rawLineage] = await Promise.all([
    readWorkspaceQuestion(workspaceId),
    fetchRootFlowRunLineage(candidateRootFlowRunIds),
  ]);
  const lineage = sortLineageEntries(rawLineage, candidateRootFlowRunIds);
  const rootFlowRunIds = lineage.map((entry) => entry.rootFlowRunId);
  const stagesResolved = await buildStageRuns(rootFlowRunIds, lineage);
  const bootstrapQuestion =
    lineage
      .slice()
      .reverse()
      .find((entry) => entry.query)?.query ?? undefined;

  const createdAt =
    lineage.find((entry) => entry.createdAt)?.createdAt ?? new Date(0).toISOString();
  const stagesWithPersistedArtifacts = await mergePersistedStageRuns(
    workspaceId,
    stagesResolved,
    createdAt,
  );

  return {
    workspaceId,
    createdAt,
    question: bootstrapQuestion ?? storedQuestion,
    rootFlowRunIds,
    latestRootFlowRunId: rootFlowRunIds.at(-1) ?? null,
    stages: stagesWithPersistedArtifacts,
  };
}
