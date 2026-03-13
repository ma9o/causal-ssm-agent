import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

export const DATA_DIR = join(process.cwd(), "..", "..", "data");
export const SESSIONS_PATH = join(DATA_DIR, "sessions.json");
export const SESSIONS_SEED_PATH = join(DATA_DIR, "sessions.seed.json");

export interface Session {
  createdAt: string;
  flowRunId?: string;
}

/** Session enriched with the question read from ``data/{userId}/query.txt``. */
export interface SessionWithQuestion extends Session {
  question?: string;
}

export async function readSessions(): Promise<Record<string, Session>> {
  // Merge tracked seed (fixture sessions) with runtime sessions.json
  let sessions: Record<string, Session> = {};
  try {
    const seed = await readFile(SESSIONS_SEED_PATH, "utf-8");
    sessions = { ...sessions, ...JSON.parse(seed) };
  } catch {
    // No seed file
  }
  try {
    const data = await readFile(SESSIONS_PATH, "utf-8");
    sessions = { ...sessions, ...JSON.parse(data) };
  } catch {
    // No runtime sessions
  }
  return sessions;
}

export async function writeSessions(sessions: Record<string, Session>): Promise<void> {
  await mkdir(dirname(SESSIONS_PATH), { recursive: true });
  await writeFile(SESSIONS_PATH, JSON.stringify(sessions, null, 2));
}

/** Read the research question from ``data/{userId}/query.txt``. */
export async function readQuestion(userId: string): Promise<string | undefined> {
  try {
    const text = await readFile(join(DATA_DIR, userId, "query.txt"), "utf-8");
    return text.trim() || undefined;
  } catch {
    return undefined;
  }
}
