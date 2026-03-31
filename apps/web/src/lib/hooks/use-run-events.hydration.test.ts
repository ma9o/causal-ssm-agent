import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { STAGES, type StageId } from "@causal-ssm/api-types";
import type { AnalysisStageRun, AnalysisStageRuns } from "@/lib/api/analysis";
import { getStage4StateQueryKey } from "@/lib/stage4-runtime";
import { createElement } from "react";
import TestRenderer, { act, type ReactTestRenderer } from "react-test-renderer";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useRunEvents, type PipelineProgress } from "./use-run-events";

vi.mock("./use-prefect-socket", () => ({
  usePrefectSocketSubscription: () => "idle",
}));

function emptyStageRun(): AnalysisStageRun {
  return {
    ownerRootFlowRunId: null,
    stageSubflowRunId: null,
    initialLogFlowRunIds: [],
    execution: null,
  };
}

function makeStageRuns(stageId: StageId, stateType: string): AnalysisStageRuns {
  const stageRuns = Object.fromEntries(
    STAGES.map((stage) => [stage.id, emptyStageRun()]),
  ) as AnalysisStageRuns;

  stageRuns[stageId] = {
    ownerRootFlowRunId: "root-flow-run",
    stageSubflowRunId: "stage-subflow-run",
    initialLogFlowRunIds: ["stage-subflow-run"],
    execution: {
      stateType,
      startTime: "2026-03-26T17:00:00.000Z",
      endTime: stateType === "RUNNING" ? null : "2026-03-26T17:10:00.000Z",
    },
  };

  return stageRuns;
}

function ProgressProbe({
  rootFlowRunIds,
  stageRuns,
  workspaceId,
}: {
  rootFlowRunIds: string[];
  stageRuns: AnalysisStageRuns;
  workspaceId: string;
}) {
  useRunEvents(workspaceId, rootFlowRunIds, stageRuns);
  return null;
}

class RunEventsTestDriver {
  readonly queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  renderer: ReactTestRenderer | null = null;

  async render({
    rootFlowRunIds,
    stageRuns,
    workspaceId,
  }: {
    rootFlowRunIds: string[];
    stageRuns: AnalysisStageRuns;
    workspaceId: string;
  }) {
    const tree = createElement(
      QueryClientProvider,
      { client: this.queryClient },
      createElement(ProgressProbe, {
        rootFlowRunIds,
        stageRuns,
        workspaceId,
      }),
    );

    await act(async () => {
      if (this.renderer) {
        this.renderer.update(tree);
      } else {
        this.renderer = TestRenderer.create(tree);
      }
    });
  }

  readProgress(workspaceId: string): PipelineProgress | undefined {
    return this.queryClient.getQueryData(["pipeline", workspaceId, "status"]);
  }

  async dispose() {
    if (this.renderer) {
      await act(async () => {
        this.renderer?.unmount();
      });
      this.renderer = null;
    }
    this.queryClient.clear();
  }
}

afterEach(() => {
  vi.clearAllMocks();
});

beforeEach(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
});

describe("useRunEvents manifest hydration", () => {
  it("rehydrates pipeline progress when manifest stage execution changes under the same lineage", async () => {
    const driver = new RunEventsTestDriver();
    const workspaceId = "SMALLGOLDEN";

    try {
      await driver.render({
        workspaceId,
        rootFlowRunIds: ["root-flow-run"],
        stageRuns: makeStageRuns("stage-4", "RUNNING"),
      });

      expect(driver.readProgress(workspaceId)?.stages["stage-4"]).toBe("running");
      expect(driver.readProgress(workspaceId)?.isFailed).toBe(false);

      await driver.render({
        workspaceId,
        rootFlowRunIds: ["root-flow-run"],
        stageRuns: makeStageRuns("stage-4", "FAILED"),
      });

      expect(driver.readProgress(workspaceId)?.stages["stage-4"]).toBe("failed");
      expect(driver.readProgress(workspaceId)?.isFailed).toBe(true);
    } finally {
      await driver.dispose();
    }
  });

  it("clears cached Stage 4 replay state when the active lineage changes", async () => {
    const driver = new RunEventsTestDriver();
    const workspaceId = "SMALLGOLDEN";

    try {
      await driver.render({
        workspaceId,
        rootFlowRunIds: ["old-root"],
        stageRuns: makeStageRuns("stage-4", "RUNNING"),
      });

      driver.queryClient.setQueryData(getStage4StateQueryKey(workspaceId, "old-root"), {
        graph: { nodes: [], edges: [], phases: [] },
        snapshot: null,
      });
      expect(
        driver.queryClient.getQueryData(getStage4StateQueryKey(workspaceId, "old-root")),
      ).toBeTruthy();

      await driver.render({
        workspaceId,
        rootFlowRunIds: ["new-root"],
        stageRuns: makeStageRuns("stage-4", "RUNNING"),
      });

      expect(
        driver.queryClient.getQueryData(getStage4StateQueryKey(workspaceId, "old-root")),
      ).toBeUndefined();
    } finally {
      await driver.dispose();
    }
  });
});
