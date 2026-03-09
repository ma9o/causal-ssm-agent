import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { defineTool } from "../define-tool";
import { uploadDataFile } from "../services/file-upload";
import { getDeploymentId, triggerRun } from "../services/prefect";

const __dirname = dirname(fileURLToPath(import.meta.url));

const CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";
const CODE_LENGTH = 6;

function generateSessionCode(): string {
  const values = crypto.getRandomValues(new Uint8Array(CODE_LENGTH));
  return Array.from(values, (v) => CHARSET[v % CHARSET.length]).join("");
}

const SESSIONS_PATH =
  process.env.SESSIONS_PATH ??
  resolve(__dirname, "..", "..", "..", "data-pipeline", "results", "sessions.json");

async function registerSession(code: string, runId: string, question: string): Promise<void> {
  let sessions: Record<string, unknown> = {};
  try {
    sessions = JSON.parse(await readFile(SESSIONS_PATH, "utf-8"));
  } catch {
    // File doesn't exist yet
  }
  sessions[code] = { runId, question, createdAt: new Date().toISOString() };
  await writeFile(SESSIONS_PATH, JSON.stringify(sessions, null, 2));
}

interface AnalyzeArgs {
  question: string;
  data_path: string;
  override_gates?: boolean;
}

export function registerAnalyzeTool(server: McpServer) {
  defineTool(
    server,
    "analyze",
    "Start a causal inference pipeline. Provide a research question and path to a .zip data file.",
    {
      question: z.string(),
      data_path: z.string(),
      override_gates: z.boolean().optional(),
    },
    async (raw) => {
      const args = raw as unknown as AnalyzeArgs;
      const question = args.question;
      const data_path = args.data_path;
      const override_gates = args.override_gates ?? false;
      const sessionCode = generateSessionCode();

      await uploadDataFile(data_path, sessionCode);

      const deploymentId = await getDeploymentId();
      const runId = await triggerRun(deploymentId, {
        query: question,
        user_id: sessionCode,
        override_gates,
      });

      await registerSession(sessionCode, runId, question);

      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              run_id: runId,
              session_code: sessionCode,
              message: "Pipeline started. Use results() to check progress.",
            }),
          },
        ],
      };
    },
  );
}
