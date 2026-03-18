import type {
  AnalysisManifest,
  AnalysisStageRun,
  AnalysisStageRuns,
  AnalysisStageTaskRun,
} from "@/lib/api/analysis";
import { dedupeRootFlowRunIds } from "@/lib/root-flow-runs";
import { STAGES, type StageId } from "@causal-ssm/api-types";
import {
  getLatestSessionRootFlowRunId,
  readQuestion,
  readSessions,
} from "../sessions/_shared";
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
}

function emptyStageRun(): AnalysisStageRun {
  return {
    ownerRootFlowRunId: null,
    stageSubflowRunId: null,
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
  const response = await fetch(`${PREFECT_API}${path}`, { cache: "no-store" });
  if (!response.ok) return null;
  return response.json();
}

async function prefectPostJson<T>(path: string, body: unknown): Promise<T | null> {
  const response = await fetch(`${PREFECT_API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) return null;
  return response.json();
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
      };
    }),
  );
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
            wrapperTaskRun: null,
          },
        ] as const;
      }

      return [
        stage.id,
        {
          ownerRootFlowRunId,
          stageSubflowRunId: await fetchStageSubflowRunId(stage.id, wrapperTaskRun.id),
          wrapperTaskRun: summarizeTaskRun(wrapperTaskRun),
        },
      ] as const;
    }),
  );

  return Object.fromEntries(stageRuns) as AnalysisStageRuns;
}

export async function buildAnalysisManifest(
  userId: string,
  bootstrapRootFlowRunIds: string[] = [],
): Promise<AnalysisManifest | null> {
  const sessions = await readSessions();
  const session = sessions[userId];
  const rootFlowRunIds = dedupeRootFlowRunIds([
    ...(session?.rootFlowRunIds ?? []),
    ...bootstrapRootFlowRunIds,
  ]);
  if (!session && rootFlowRunIds.length === 0) return null;

  const [question, lineage] = await Promise.all([
    session ? readQuestion(userId) : Promise.resolve(undefined),
    fetchRootFlowRunLineage(rootFlowRunIds),
  ]);
  const stagesResolved = await buildStageRuns(rootFlowRunIds, lineage);

  const createdAt =
    session?.createdAt ??
    lineage.find((entry) => entry.createdAt)?.createdAt ??
    new Date(0).toISOString();

  return {
    userId,
    createdAt,
    question,
    rootFlowRunIds,
    latestRootFlowRunId: rootFlowRunIds.at(-1) ?? getLatestSessionRootFlowRunId(session),
    stages: stagesResolved,
  };
}
