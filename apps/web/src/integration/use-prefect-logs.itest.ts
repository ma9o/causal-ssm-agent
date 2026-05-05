import { QueryClient, QueryClientProvider, type QueryStatus } from "@tanstack/react-query";
import { randomUUID } from "node:crypto";
import { createElement, useEffect } from "react";
import TestRenderer, { act, type ReactTestRenderer } from "react-test-renderer";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { usePrefectLogs } from "@/lib/hooks/use-stage-logs";
import type { StageRunStatus } from "@/lib/hooks/pipeline-progress";
import type { PrefectSocketConnectionState } from "@/lib/hooks/use-prefect-socket";
import { buildPrefectSubscriptionKey } from "@/lib/stage-observability";
import {
  delay,
  emitLogs,
  insertPersistedLogs,
  startPrefectServer,
  stopPrefectServer,
  type PrefectServerHandle,
} from "./prefect-test-harness";

const WAIT_TIMEOUT_MS = 10_000;

type PrefectLogsSnapshot = {
  bootstrapStatus: QueryStatus;
  connectionState: PrefectSocketConnectionState;
  messages: string[];
};

type PrefectLogsProbeProps = {
  flowRunIds: string[];
  onSnapshot: (snapshot: PrefectLogsSnapshot) => void;
  queryKey: readonly unknown[];
  status: StageRunStatus;
};

function PrefectLogsProbe({
  flowRunIds,
  onSnapshot,
  queryKey,
  status,
}: PrefectLogsProbeProps) {
  const { bootstrapStatus, connectionState, logs } = usePrefectLogs(
    queryKey,
    flowRunIds,
    {},
    buildPrefectSubscriptionKey(flowRunIds),
    status,
  );

  useEffect(() => {
    onSnapshot({
      bootstrapStatus,
      connectionState,
      messages: logs.map((entry) => entry.message),
    });
  }, [bootstrapStatus, connectionState, logs, onSnapshot]);

  return null;
}

class PrefectLogsTestDriver {
  private readonly queryClient: QueryClient;
  private renderer: ReactTestRenderer | null = null;
  private snapshot: PrefectLogsSnapshot | null = null;

  constructor() {
    this.queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
  }

  getSnapshot() {
    return this.snapshot;
  }

  async render({
    flowRunIds,
    queryKey,
    status,
  }: {
    flowRunIds: string[];
    queryKey: readonly unknown[];
    status: StageRunStatus;
  }) {
    const tree = createElement(
      QueryClientProvider,
      { client: this.queryClient },
      createElement(PrefectLogsProbe, {
        flowRunIds,
        onSnapshot: (snapshot: PrefectLogsSnapshot) => {
          this.snapshot = snapshot;
        },
        queryKey,
        status,
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

async function waitForSnapshot(
  driver: PrefectLogsTestDriver,
  predicate: (snapshot: PrefectLogsSnapshot) => boolean,
  timeoutMs = WAIT_TIMEOUT_MS,
): Promise<PrefectLogsSnapshot> {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    const snapshot = driver.getSnapshot();
    if (snapshot && predicate(snapshot)) {
      return snapshot;
    }

    await act(async () => {
      await delay(50);
    });
  }

  throw new Error("Timed out waiting for Prefect logs snapshot");
}

describe("usePrefectLogs integration", () => {
  let server: PrefectServerHandle | null = null;
  let originalFetch: typeof globalThis.fetch;
  let originalPrefectLogsUrl: string | undefined;
  let originalActEnvironment: boolean | undefined;
  let originalWindow: typeof globalThis.window | undefined;
  const drivers = new Set<PrefectLogsTestDriver>();

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
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;

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

  it(
    "loads persisted logs on the running-to-failed transition even when the websocket never delivered them",
    async () => {
      if (!server) {
        throw new Error("Prefect server did not start");
      }

      const flowRunId = randomUUID();
      const driver = new PrefectLogsTestDriver();
      drivers.add(driver);
      const queryKey = ["itest", "terminal-catch-up", flowRunId] as const;

      await driver.render({
        flowRunIds: [flowRunId],
        queryKey,
        status: "running",
      });

      await waitForSnapshot(
        driver,
        (snapshot) =>
          snapshot.bootstrapStatus === "success" && snapshot.connectionState === "streaming",
      );

      await emitLogs(server.apiBaseUrl, flowRunId, ["live-1"]);
      await waitForSnapshot(driver, (snapshot) => snapshot.messages.includes("live-1"));

      insertPersistedLogs(server.dbPath, flowRunId, ["missed-1", "missed-2"]);

      await driver.render({
        flowRunIds: [flowRunId],
        queryKey,
        status: "failed",
      });

      const snapshot = await waitForSnapshot(
        driver,
        (next) =>
          next.connectionState === "idle" &&
          next.messages.length === 3 &&
          next.messages.includes("missed-1") &&
          next.messages.includes("missed-2"),
      );

      expect(snapshot.messages).toEqual(["live-1", "missed-1", "missed-2"]);
    },
    30_000,
  );

  it(
    "replays persisted logs for newly added flow runs when the subscription scope widens",
    async () => {
      if (!server) {
        throw new Error("Prefect server did not start");
      }

      const primaryFlowRunId = randomUUID();
      const addedFlowRunId = randomUUID();
      const driver = new PrefectLogsTestDriver();
      drivers.add(driver);
      const queryKey = ["itest", "scope-widening", primaryFlowRunId] as const;

      await driver.render({
        flowRunIds: [primaryFlowRunId],
        queryKey,
        status: "running",
      });

      await waitForSnapshot(
        driver,
        (snapshot) =>
          snapshot.bootstrapStatus === "success" && snapshot.connectionState === "streaming",
      );

      await emitLogs(server.apiBaseUrl, primaryFlowRunId, ["primary-live-1"]);
      await waitForSnapshot(driver, (snapshot) => snapshot.messages.includes("primary-live-1"));

      insertPersistedLogs(server.dbPath, addedFlowRunId, ["added-1", "added-2"]);

      await driver.render({
        flowRunIds: [primaryFlowRunId, addedFlowRunId],
        queryKey,
        status: "running",
      });

      const snapshot = await waitForSnapshot(
        driver,
        (next) =>
          next.messages.length === 3 &&
          next.messages.includes("primary-live-1") &&
          next.messages.includes("added-1") &&
          next.messages.includes("added-2"),
      );

      expect(snapshot.messages).toEqual(["primary-live-1", "added-1", "added-2"]);
    },
    30_000,
  );
});
