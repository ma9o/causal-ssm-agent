import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { randomUUID } from "node:crypto";
import { createElement } from "react";
import TestRenderer, {
  act,
  type ReactTestInstance,
  type ReactTestRenderer,
} from "react-test-renderer";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { StageSectionRouter } from "@/components/pipeline/stage-section-router";
import { RefinementProvider } from "@/lib/contexts/refinement-context";
import type { AnalysisStageExecution, AnalysisStageRun } from "@/lib/api/analysis";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { STAGES } from "@nof1-causal-lab/api-types";
import {
  delay,
  emitLogs,
  insertPersistedLogs,
  startPrefectServer,
  stopPrefectServer,
  type PrefectServerHandle,
} from "./prefect-test-harness";

const WAIT_TIMEOUT_MS = 10_000;
function getStage4Meta() {
  const stage = STAGES.find((candidate) => candidate.id === "stage-4");
  if (!stage) {
    throw new Error("Stage 4 metadata is unavailable");
  }
  return stage;
}

const STAGE = getStage4Meta();

function normalizeText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function collectInstanceText(value: ReactTestInstance | string): string {
  if (typeof value === "string") {
    return value;
  }

  return value.children.map((child) => collectInstanceText(child)).join("");
}

function buildStageExecution(stateType: string): AnalysisStageExecution {
  const now = new Date().toISOString();
  return {
    stateType,
    startTime: now,
    endTime: stateType === "RUNNING" ? null : now,
  };
}

function buildStageRun(flowRunId: string, executionStateType: string): AnalysisStageRun {
  return {
    ownerRootFlowRunId: "root-flow-run",
    stageSubflowRunId: flowRunId,
    initialLogFlowRunIds: [flowRunId],
    execution: buildStageExecution(executionStateType),
  };
}

class StageSectionRouterTestDriver {
  private readonly queryClient: QueryClient;
  private renderer: ReactTestRenderer | null = null;

  constructor() {
    this.queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
  }

  async render({ stageRun, status }: { stageRun: AnalysisStageRun; status: StageRunStatus }) {
    const tree = createElement(
      QueryClientProvider,
      { client: this.queryClient },
      createElement(
        RefinementProvider,
        null,
        createElement(StageSectionRouter, {
          stage: STAGE,
          stageRun,
          status,
          timing: undefined,
          workspaceId: "integration-workspace",
        }),
      ),
    );

    await act(async () => {
      if (this.renderer) {
        this.renderer.update(tree);
      } else {
        this.renderer = TestRenderer.create(tree);
      }
    });
  }

  readLogButtonText(): string {
    if (!this.renderer) {
      throw new Error("StageSectionRouter has not been rendered");
    }

    const buttons = this.renderer.root.findAllByType("button");
    if (buttons.length === 0) {
      throw new Error("Could not find the stage log button");
    }

    return normalizeText(collectInstanceText(buttons[buttons.length - 1]));
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

async function waitForButtonText(
  driver: StageSectionRouterTestDriver,
  predicate: (text: string) => boolean,
  timeoutMs = WAIT_TIMEOUT_MS,
): Promise<string> {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    const text = driver.readLogButtonText();
    if (predicate(text)) {
      return text;
    }

    await act(async () => {
      await delay(50);
    });
  }

  throw new Error(
    `Timed out waiting for stage log button text; last value: ${driver.readLogButtonText()}`,
  );
}

describe("StageSectionRouter log integration", () => {
  let server: PrefectServerHandle | null = null;
  let originalFetch: typeof globalThis.fetch;
  let originalPrefectLogsUrl: string | undefined;
  let originalActEnvironment: boolean | undefined;
  let originalWindow: typeof globalThis.window | undefined;
  const drivers = new Set<StageSectionRouterTestDriver>();

  beforeAll(async () => {
    server = await startPrefectServer();
    originalFetch = globalThis.fetch;
    originalPrefectLogsUrl = process.env.NEXT_PUBLIC_PREFECT_LOGS_URL;
    originalActEnvironment = (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean })
      .IS_REACT_ACT_ENVIRONMENT;
    originalWindow = globalThis.window;

    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    globalThis.window = {
      location: { origin: "http://localhost:3000" },
    } as typeof globalThis.window;
    process.env.NEXT_PUBLIC_PREFECT_LOGS_URL = server.wsUrl;

    globalThis.fetch = (input, init) => {
      const url =
        typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;

      if (url === "/prefect/logs/filter") {
        return originalFetch(`${server?.apiBaseUrl}/logs/filter`, init);
      }

      return originalFetch(input as RequestInfo | URL, init);
    };
  }, 60_000);

  afterEach(async () => {
    for (const driver of drivers) {
      await driver.dispose();
    }
    drivers.clear();
  });

  afterAll(async () => {
    globalThis.fetch = originalFetch;

    if (originalPrefectLogsUrl === undefined) {
      delete process.env.NEXT_PUBLIC_PREFECT_LOGS_URL;
    } else {
      process.env.NEXT_PUBLIC_PREFECT_LOGS_URL = originalPrefectLogsUrl;
    }

    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
      originalActEnvironment;

    if (originalWindow === undefined) {
      delete (globalThis as { window?: typeof globalThis.window }).window;
    } else {
      globalThis.window = originalWindow;
    }

    if (server) {
      await stopPrefectServer(server);
    }
  }, 60_000);

  it("catches up when manifest turns terminal before progress catches up", async () => {
    if (!server) {
      throw new Error("Prefect server did not start");
    }

    const flowRunId = randomUUID();
    const driver = new StageSectionRouterTestDriver();
    drivers.add(driver);

    await driver.render({
      stageRun: buildStageRun(flowRunId, "RUNNING"),
      status: "running",
    });

    await waitForButtonText(driver, (text) => text.includes("Show logs"));

    await emitLogs(server.apiBaseUrl, flowRunId, ["live-1"]);
    await waitForButtonText(driver, (text) => text.includes("(1)"));

    insertPersistedLogs(server.dbPath, flowRunId, ["missed-1", "missed-2"]);

    await driver.render({
      stageRun: buildStageRun(flowRunId, "FAILED"),
      status: "running",
    });

    const intermediate = await waitForButtonText(driver, (text) => text.includes("(1)"));
    expect(intermediate).toContain("Show logs");

    await driver.render({
      stageRun: buildStageRun(flowRunId, "FAILED"),
      status: "failed",
    });

    const terminal = await waitForButtonText(driver, (text) => text.includes("(3)"));
    expect(terminal).toContain("Show logs");
  }, 30_000);

  it("catches up when progress turns terminal before the manifest updates", async () => {
    if (!server) {
      throw new Error("Prefect server did not start");
    }

    const flowRunId = randomUUID();
    const driver = new StageSectionRouterTestDriver();
    drivers.add(driver);

    await driver.render({
      stageRun: buildStageRun(flowRunId, "RUNNING"),
      status: "running",
    });

    await emitLogs(server.apiBaseUrl, flowRunId, ["live-1"]);
    await waitForButtonText(driver, (text) => text.includes("(1)"));

    insertPersistedLogs(server.dbPath, flowRunId, ["missed-1", "missed-2"]);

    await driver.render({
      stageRun: buildStageRun(flowRunId, "RUNNING"),
      status: "failed",
    });

    const fromProgress = await waitForButtonText(driver, (text) => text.includes("(3)"));
    expect(fromProgress).toContain("Show logs");

    await driver.render({
      stageRun: buildStageRun(flowRunId, "FAILED"),
      status: "failed",
    });

    const settled = await waitForButtonText(driver, (text) => text.includes("(3)"));
    expect(settled).toContain("Show logs");
  }, 30_000);
});
