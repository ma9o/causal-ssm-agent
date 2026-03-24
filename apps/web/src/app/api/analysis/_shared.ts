import type {
  AnalysisManifest,
  AnalysisStageRun,
  AnalysisStageRuns,
  AnalysisStageTaskRun,
} from "@/lib/api/analysis";
import { dedupeRootFlowRunIds, getWorkspaceRunTag } from "@/lib/root-flow-runs";
import { prefectFetch } from "@/lib/server/prefect-runs";
import { readData } from "@/lib/storage";
import { STAGES, type StageId } from "@causal-ssm/api-types";
import { getPrefectApiUrl } from "@/lib/runtime-urls";

const PREFECT_API = getPrefectApiUrl();

interface PrefectFlowRun {
  id: string;
  parameters?: Record<string, unknown>;
  created?: string | null;
  start_time?: string | null;
  expected_start_time?: string | null;
}

interface PrefectTaskRun {
  id: string;
  name: string;
  state_type: string;
  start_time: string | null;
  end_time: string | null;
}

interface RootFlowRunLineageEntry {
  rootFlowRunId: string;
  startStage: StageId;
  endStage: StageId;
  createdAt: string | null;
  query: string | null;
}

function emptyStageRun(): AnalysisStageRun {
  return {
    ownerRootFlowRunId: null,
    stageSubflowRunId: null,
    logFlowRunIds: [],
    wrapperTaskRun: null,
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
  stageId: StageId,
): string | null {
  const stageIndex = getStageIndex(stageId);
  if (stageIndex === -1) return null;

  let ownerRootFlowRunId: string | null = null;

  for (const entry of lineage) {
    const startIndex = getStageIndex(entry.startStage);
    const endIndex = getStageIndex(entry.endStage);
    if (startIndex <= stageIndex && stageIndex <= endIndex) {
      ownerRootFlowRunId = entry.rootFlowRunId;
    }
  }

  return ownerRootFlowRunId;
}

function summarizeTaskRun(taskRun: PrefectTaskRun): AnalysisStageTaskRun {
  return {
    id: taskRun.id,
    name: taskRun.name,
    stateType: taskRun.state_type,
    startTime: taskRun.start_time,
    endTime: taskRun.end_time,
  };
}

async function prefectGetJson<T>(path: string): Promise<T | null> {
  const response = await prefectFetch(`${PREFECT_API}${path}`, { cache: "no-store" });
  if (!response.ok) return null;
  return response.json();
}

async function prefectPostJson<T>(path: string, body: unknown): Promise<T | null> {
  const response = await prefectFetch(`${PREFECT_API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) return null;
  return response.json();
}

async function fetchWorkspaceRootFlowRunIds(workspaceId: string): Promise<string[]> {
  const flowRuns =
    (await prefectPostJson<PrefectFlowRun[]>("/flow_runs/filter", {
      flow_runs: {
        tags: { all_: [getWorkspaceRunTag(workspaceId)] },
        parent_task_run_id: { is_null_: true },
      },
      sort: "START_TIME_DESC",
      limit: 200,
    })) ?? [];

  return flowRuns.map((flowRun) => flowRun.id);
}

async function readWorkspaceQuestion(workspaceId: string): Promise<string | undefined> {
  try {
    const text = await readData(`${workspaceId}/query.txt`);
    const trimmed = text.trim();
    return trimmed || undefined;
  } catch {
    return undefined;
  }
}

async function fetchRootFlowRunLineage(
  rootFlowRunIds: string[],
): Promise<RootFlowRunLineageEntry[]> {
  return Promise.all(
    rootFlowRunIds.map(async (rootFlowRunId) => {
      const flowRun = await prefectGetJson<PrefectFlowRun>(`/flow_runs/${rootFlowRunId}`);
      return {
        rootFlowRunId,
        startStage: getFlowRunStartStage(flowRun?.parameters),
        endStage: getFlowRunEndStage(flowRun?.parameters),
        createdAt: flowRun?.created ?? flowRun?.start_time ?? flowRun?.expected_start_time ?? null,
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

async function fetchTaskRunsForRootFlowRun(rootFlowRunId: string): Promise<PrefectTaskRun[]> {
  return (
    (await prefectPostJson<PrefectTaskRun[]>("/task_runs/filter", {
      flow_runs: { id: { any_: [rootFlowRunId] } },
      sort: "EXPECTED_START_TIME_DESC",
    })) ?? []
  );
}

function findStageWrapperTaskRun(
  taskRuns: PrefectTaskRun[],
  stageId: StageId,
): PrefectTaskRun | null {
  const stage = STAGES.find((candidate) => candidate.id === stageId);
  if (!stage) return null;

  return (
    taskRuns.find((candidate) =>
      candidate.name === stage.prefectFlowName ||
      candidate.name.startsWith(`${stage.prefectFlowName}-`),
    ) ?? null
  );
}

async function fetchStageSubflowRunId(stageId: StageId, parentTaskRunId: string): Promise<string | null> {
  const stage = STAGES.find((candidate) => candidate.id === stageId);
  if (!stage) return null;

  const flowRuns = await prefectPostJson<PrefectFlowRun[]>("/flow_runs/filter", {
    flows: { name: { any_: [stage.prefectFlowName] } },
    flow_runs: { parent_task_run_id: { any_: [parentTaskRunId] } },
    sort: "START_TIME_DESC",
    limit: 1,
  });

  return flowRuns?.[0]?.id ?? null;
}

export async function fetchStageLogFlowRunIds(
  stageId: StageId,
  stageSubflowRunId: string | null,
): Promise<string[]> {
  if (!stageSubflowRunId) {
    return [];
  }

  if (stageId !== "stage-2") {
    return [stageSubflowRunId];
  }

  const childFlowRuns =
    (await prefectPostJson<PrefectFlowRun[]>("/flow_runs/filter", {
      flow_runs: { parent_flow_run_id: { any_: [stageSubflowRunId] } },
      sort: "START_TIME_ASC",
      limit: 50,
    })) ?? [];

  return [...new Set([stageSubflowRunId, ...childFlowRuns.map((flowRun) => flowRun.id)])];
}

async function buildStageRuns(
  rootFlowRunIds: string[],
  lineage?: RootFlowRunLineageEntry[],
): Promise<AnalysisStageRuns> {
  const effectiveLineage = lineage ?? (await fetchRootFlowRunLineage(rootFlowRunIds));
  const taskRunsByRootFlowRunId = new Map(
    await Promise.all(
      effectiveLineage.map(async ({ rootFlowRunId }) => [
        rootFlowRunId,
        await fetchTaskRunsForRootFlowRun(rootFlowRunId),
      ] as const),
    ),
  );

  const stageRuns = await Promise.all(
    STAGES.map(async (stage) => {
      const ownerRootFlowRunId = getStageOwningRootFlowRunId(effectiveLineage, stage.id);
      if (!ownerRootFlowRunId) {
        return [stage.id, emptyStageRun()] as const;
      }

      const wrapperTaskRun = findStageWrapperTaskRun(
        taskRunsByRootFlowRunId.get(ownerRootFlowRunId) ?? [],
        stage.id,
      );

      if (!wrapperTaskRun) {
        return [
          stage.id,
          {
            ownerRootFlowRunId,
            stageSubflowRunId: null,
            logFlowRunIds: [],
            wrapperTaskRun: null,
          },
        ] as const;
      }

      const stageSubflowRunId = await fetchStageSubflowRunId(stage.id, wrapperTaskRun.id);

      return [
        stage.id,
        {
          ownerRootFlowRunId,
          stageSubflowRunId,
          logFlowRunIds: await fetchStageLogFlowRunIds(stage.id, stageSubflowRunId),
          wrapperTaskRun: summarizeTaskRun(wrapperTaskRun),
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
  const prefectRootFlowRunIds = await fetchWorkspaceRootFlowRunIds(workspaceId);
  const candidateRootFlowRunIds = dedupeRootFlowRunIds([
    ...prefectRootFlowRunIds,
    ...bootstrapRootFlowRunIds,
  ]);
  if (candidateRootFlowRunIds.length === 0) return null;

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

  return {
    workspaceId,
    createdAt,
    question: bootstrapQuestion ?? storedQuestion,
    rootFlowRunIds,
    latestRootFlowRunId: rootFlowRunIds.at(-1) ?? null,
    stages: stagesResolved,
  };
}
