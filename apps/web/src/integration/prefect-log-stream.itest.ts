import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { randomUUID } from "node:crypto";
import {
  buildPrefectLogStreamFilterMessage,
  fetchIncrementalPrefectLogs,
  mergePrefectLogs,
  type PrefectLogEntry,
} from "@/lib/prefect-log-client";
import {
  delay,
  emitLogs,
  startPrefectServer,
  stopPrefectServer,
  type PrefectServerHandle,
} from "./prefect-test-harness";

const LOG_WAIT_TIMEOUT_MS = 10_000;

class PrefectLogConsumer {
  private readonly flowRunIds: string[];
  private readonly wsUrl: string;
  private socket: WebSocket | null = null;
  logs: PrefectLogEntry[] = [];

  constructor(flowRunIds: string[], wsUrl: string) {
    this.flowRunIds = flowRunIds;
    this.wsUrl = wsUrl;
  }

  async bootstrap(): Promise<void> {
    this.logs = await fetchIncrementalPrefectLogs(this.flowRunIds, [], { offset: 0 });
  }

  async catchUp(offset?: number): Promise<void> {
    this.logs = await fetchIncrementalPrefectLogs(this.flowRunIds, this.logs, {
      ...(offset === undefined ? {} : { offset }),
    });
  }

  async connect(): Promise<void> {
    await this.close();

    const socket = new WebSocket(this.wsUrl, "prefect");
    this.socket = socket;

    await new Promise<void>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error("Timed out waiting for Prefect log socket auth"));
      }, LOG_WAIT_TIMEOUT_MS);

      const cleanup = () => {
        clearTimeout(timeoutId);
        socket.onopen = null;
        socket.onerror = null;
      };

      socket.onopen = () => {
        socket.send(JSON.stringify({ type: "auth", token: null }));
      };

      socket.onerror = () => {
        cleanup();
        reject(new Error("Prefect log socket failed before auth"));
      };

      socket.onmessage = (event: MessageEvent) => {
        const message = JSON.parse(String(event.data)) as {
          type?: string;
          log?: PrefectLogEntry;
        };

        if (message.type === "auth_success") {
          socket.send(JSON.stringify(buildPrefectLogStreamFilterMessage(this.flowRunIds)));
          cleanup();
          resolve();
          return;
        }

        if (message.type === "auth_failure") {
          cleanup();
          reject(new Error("Prefect log socket auth failed"));
          return;
        }

        if (message.type === "log" && message.log) {
          this.logs = mergePrefectLogs(this.logs, [message.log]);
        }
      };
    });
  }

  async close(): Promise<void> {
    if (!this.socket) {
      return;
    }

    const socket = this.socket;
    this.socket = null;

    await new Promise<void>((resolve) => {
      if (socket.readyState === WebSocket.CLOSED) {
        resolve();
        return;
      }

      socket.onclose = () => resolve();
      socket.close();
    });
  }
}

async function waitForLogCount(
  consumer: PrefectLogConsumer,
  count: number,
  timeoutMs = LOG_WAIT_TIMEOUT_MS,
) {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    if (consumer.logs.length >= count) {
      return;
    }
    await delay(50);
  }

  throw new Error(`Timed out waiting for ${count} logs; saw ${consumer.logs.length}`);
}

describe("Prefect log streaming integration", () => {
  let server: PrefectServerHandle | null = null;
  let originalFetch: typeof globalThis.fetch;
  const consumers = new Set<PrefectLogConsumer>();

  beforeAll(async () => {
    server = await startPrefectServer();
    originalFetch = globalThis.fetch;

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
    for (const consumer of consumers) {
      await consumer.close();
    }
    consumers.clear();
  });

  afterAll(async () => {
    globalThis.fetch = originalFetch;
    if (server) {
      await stopPrefectServer(server);
    }
  }, 60_000);

  it("shows that Prefect logs/out is live-only and requires REST catch-up for older logs", async () => {
    if (!server) {
      throw new Error("Prefect server did not start");
    }
    const flowRunId = randomUUID();
    const consumer = new PrefectLogConsumer([flowRunId], server.wsUrl);
    consumers.add(consumer);

    await emitLogs(server.apiBaseUrl, flowRunId, ["before-1", "before-2"]);
    await consumer.bootstrap();
    expect(consumer.logs.map((entry) => entry.message)).toEqual(["before-1", "before-2"]);

    consumer.logs = [];
    await consumer.connect();
    await delay(250);
    expect(consumer.logs).toHaveLength(0);

    await emitLogs(server.apiBaseUrl, flowRunId, ["after-1", "after-2"]);
    await waitForLogCount(consumer, 2);
    expect(consumer.logs.map((entry) => entry.message)).toEqual(["after-1", "after-2"]);

    await consumer.catchUp();
    expect(consumer.logs.map((entry) => entry.message)).toEqual(["after-1", "after-2"]);

    await consumer.catchUp(0);
    expect(consumer.logs).toHaveLength(4);
    expect(
      [...consumer.logs]
        .sort((left, right) => left.timestamp.localeCompare(right.timestamp))
        .map((entry) => entry.message),
    ).toEqual(["before-1", "before-2", "after-1", "after-2"]);
  }, 30_000);

  it("recovers logs emitted during a running stream gap only after an explicit catch-up", async () => {
    if (!server) {
      throw new Error("Prefect server did not start");
    }
    const flowRunId = randomUUID();
    const consumer = new PrefectLogConsumer([flowRunId], server.wsUrl);
    consumers.add(consumer);

    await consumer.bootstrap();
    expect(consumer.logs).toHaveLength(0);

    await consumer.connect();
    await emitLogs(server.apiBaseUrl, flowRunId, ["live-1"]);
    await waitForLogCount(consumer, 1);

    await consumer.close();
    await emitLogs(server.apiBaseUrl, flowRunId, ["missed-1", "missed-2"]);
    await delay(250);
    expect(consumer.logs.map((entry) => entry.message)).toEqual(["live-1"]);

    await consumer.connect();
    await delay(250);
    expect(consumer.logs.map((entry) => entry.message)).toEqual(["live-1"]);

    await consumer.catchUp();
    expect(consumer.logs.map((entry) => entry.message)).toEqual(["live-1", "missed-1", "missed-2"]);
  }, 30_000);
});
