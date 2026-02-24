import { readFile } from "node:fs/promises";
import { join } from "node:path";

export const SESSIONS_PATH = join(process.cwd(), "..", "data-pipeline", "results", "sessions.json");

export interface Session {
  runId: string;
  question: string;
  createdAt: string;
}

export async function readSessions(): Promise<Record<string, Session>> {
  try {
    const data = await readFile(SESSIONS_PATH, "utf-8");
    return JSON.parse(data);
  } catch {
    return {};
  }
}
