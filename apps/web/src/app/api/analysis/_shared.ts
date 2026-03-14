import type {
  AnalysisManifest,
  AnalysisStageRun,
  AnalysisStageRuns,
  AnalysisStageTaskRun,
} from "@/lib/api/analysis";
import { STAGES, type StageId } from "@causal-ssm/api-types";
import {
  getLatestSessionRootFlowRunId,
  readQuestion,
  readSessions,
} from "../sessions/_shared";

const PREFECT_API = "http://localhost:4200/api";

interface PrefectFlowRun {
  id: string;
  parameters?: Record<string, unknown>;
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
    if (getStageIndex(entry.startStage) <= stageIndex) {
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

async function buildStageRuns(rootFlowRunIds: string[]): Promise<AnalysisStageRuns> {
  const lineage = await fetchRootFlowRunLineage(rootFlowRunIds);
  const taskRunsByRootFlowRunId = new Map(
    await Promise.all(
      lineage.map(async ({ rootFlowRunId }) => [
        rootFlowRunId,
        await fetchTaskRunsForRootFlowRun(rootFlowRunId),
      ] as const),
    ),
  );

  const stageRuns = await Promise.all(
    STAGES.map(async (stage) => {
      const ownerRootFlowRunId = getStageOwningRootFlowRunId(lineage, stage.id);
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

export async function buildAnalysisManifest(userId: string): Promise<AnalysisManifest | null> {
  const sessions = await readSessions();
  const session = sessions[userId];
  if (!session) return null;

  const [question, stages] = await Promise.all([
    readQuestion(userId),
    buildStageRuns(session.rootFlowRunIds),
  ]);

  return {
    userId,
    createdAt: session.createdAt,
    question,
    rootFlowRunIds: session.rootFlowRunIds,
    latestRootFlowRunId: getLatestSessionRootFlowRunId(session),
    stages,
  };
}
