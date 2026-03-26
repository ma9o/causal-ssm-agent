import { spawn, spawnSync, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

const PREFECT_BOOT_TIMEOUT_MS = 30_000;
const DATA_PIPELINE_DIR = join(process.cwd(), "..", "data-pipeline");

export type PrefectServerHandle = {
  apiBaseUrl: string;
  dbPath: string;
  process: ChildProcessWithoutNullStreams;
  tempDir: string;
  wsUrl: string;
};

export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function getFreePort(): Promise<number> {
  return await new Promise<number>((resolve, reject) => {
    const server = createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("Failed to allocate a free TCP port"));
        return;
      }

      const { port } = address;
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve(port);
      });
    });
  });
}

async function waitForPrefectBoot(apiBaseUrl: string, process: ChildProcessWithoutNullStreams) {
  const startedAt = Date.now();

  while (Date.now() - startedAt < PREFECT_BOOT_TIMEOUT_MS) {
    if (process.exitCode !== null) {
      throw new Error(`Prefect server exited early with code ${process.exitCode}`);
    }

    try {
      const response = await fetch(`${apiBaseUrl}/health`);
      if (response.ok) {
        return;
      }
    } catch {
      // Ignore until the server starts responding.
    }

    await delay(250);
  }

  throw new Error("Timed out waiting for Prefect server health endpoint");
}

export async function startPrefectServer(): Promise<PrefectServerHandle> {
  const port = await getFreePort();
  const tempDir = mkdtempSync(join(tmpdir(), "prefect-log-stream-"));
  const dbPath = join(tempDir, "prefect.db");
  const apiBaseUrl = `http://127.0.0.1:${port}/api`;
  const wsUrl = `ws://127.0.0.1:${port}/api/logs/out`;

  const child = spawn(
    "uv",
    ["run", "prefect", "server", "start", "--host", "127.0.0.1", "--port", String(port), "--no-ui"],
    {
      cwd: DATA_PIPELINE_DIR,
      env: {
        ...process.env,
        PREFECT_SERVER_DATABASE_CONNECTION_URL: `sqlite+aiosqlite:///${dbPath}`,
        PREFECT_SERVER_LOGS_STREAM_OUT_ENABLED: "true",
        PREFECT_SERVER_LOGS_STREAM_PUBLISHING_ENABLED: "true",
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  child.stdout.on("data", () => {});
  child.stderr.on("data", () => {});

  await waitForPrefectBoot(apiBaseUrl, child);

  return {
    apiBaseUrl,
    dbPath,
    process: child,
    tempDir,
    wsUrl,
  };
}

export async function stopPrefectServer(handle: PrefectServerHandle): Promise<void> {
  if (handle.process.exitCode === null) {
    handle.process.kill("SIGTERM");
    await new Promise<void>((resolve) => {
      handle.process.once("exit", () => resolve());
      setTimeout(() => {
        if (handle.process.exitCode === null) {
          handle.process.kill("SIGKILL");
        }
      }, 5_000);
    });
  }

  rmSync(handle.tempDir, { force: true, recursive: true });
}

export async function emitLogs(apiBaseUrl: string, flowRunId: string, messages: string[]) {
  const now = Date.now();
  const response = await fetch(`${apiBaseUrl}/logs/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      messages.map((message, index) => ({
        name: "integration.prefect.logs",
        level: 20,
        message,
        timestamp: new Date(now + index).toISOString(),
        flow_run_id: flowRunId,
      })),
    ),
  });

  if (response.status !== 201) {
    throw new Error(`Prefect log emission failed with status ${response.status}`);
  }
}

export function insertPersistedLogs(dbPath: string, flowRunId: string, messages: string[]) {
  const payload = JSON.stringify({
    dbPath,
    flowRunId,
    messages,
    timestampBase: Date.now(),
  });

  const result = spawnSync(
    "python3",
    [
      "-c",
      `
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone

payload = json.loads(sys.stdin.read())
conn = sqlite3.connect(payload["dbPath"])
cursor = conn.cursor()

for index, message in enumerate(payload["messages"]):
    instant = datetime.fromtimestamp(
        (payload["timestampBase"] + index) / 1000,
        tz=timezone.utc,
    ).isoformat().replace("+00:00", "Z")
    cursor.execute(
        """
        INSERT INTO log (
            name,
            level,
            flow_run_id,
            task_run_id,
            message,
            timestamp,
            id,
            created,
            updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "integration.prefect.logs.persisted",
            20,
            payload["flowRunId"],
            None,
            message,
            instant,
            str(uuid.uuid4()),
            instant,
            instant,
        ),
    )

conn.commit()
conn.close()
      `,
    ],
    {
      encoding: "utf8",
      input: payload,
    },
  );

  if (result.status !== 0) {
    throw new Error(result.stderr || "Failed to insert persisted Prefect logs");
  }
}
