import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { getLatestRootFlowRunId, mergeRootFlowRunIds } from "@/lib/root-flow-runs";

export const DATA_DIR = join(process.cwd(), "..", "..", "data");
export const SESSIONS_PATH = join(DATA_DIR, "sessions.json");
export const SESSIONS_SEED_PATH = join(DATA_DIR, "sessions.seed.json");

export interface Session {
  createdAt: string;
  rootFlowRunIds: string[];
}

export function normalizeSession(session?: Session): Session {
  return {
    createdAt: session?.createdAt ?? new Date().toISOString(),
    rootFlowRunIds: mergeRootFlowRunIds(session?.rootFlowRunIds ?? []),
  };
}

export function getLatestSessionRootFlowRunId(session?: Session): string | null {
  return getLatestRootFlowRunId(session?.rootFlowRunIds ?? []);
}

export function appendSessionRootFlowRunId(
  session: Session | undefined,
  rootFlowRunId: string,
): Session {
  return {
    createdAt: session?.createdAt ?? new Date().toISOString(),
    rootFlowRunIds: mergeRootFlowRunIds(session?.rootFlowRunIds ?? [], rootFlowRunId),
  };
}

async function readSessionFile(path: string): Promise<Record<string, Session>> {
  try {
    const parsed = JSON.parse(await readFile(path, "utf-8")) as Record<string, Session>;
    return Object.fromEntries(
      Object.entries(parsed).map(([userId, session]) => [
        userId,
        normalizeSession(session),
      ]),
    );
  } catch {
    return {};
  }
}

export async function readSessions(): Promise<Record<string, Session>> {
  // Merge tracked seed (fixture sessions) with runtime sessions.json
  return {
    ...(await readSessionFile(SESSIONS_SEED_PATH)),
    ...(await readSessionFile(SESSIONS_PATH)),
  };
}

export async function writeSessions(sessions: Record<string, Session>): Promise<void> {
  await mkdir(dirname(SESSIONS_PATH), { recursive: true });
  await writeFile(
    SESSIONS_PATH,
    JSON.stringify(
      Object.fromEntries(
        Object.entries(sessions).map(([userId, session]) => [
          userId,
          normalizeSession(session),
        ]),
      ),
      null,
      2,
    ),
  );
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
