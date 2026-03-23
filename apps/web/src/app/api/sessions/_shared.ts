import { getLatestRootFlowRunId, mergeRootFlowRunIds } from "@/lib/root-flow-runs";
import { readData, writeData } from "@/lib/storage";

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

function getSessionPath(userId: string): string {
  return `${userId}/session.json`;
}

export async function readSession(userId: string): Promise<Session | null> {
  try {
    const parsed = JSON.parse(await readData(getSessionPath(userId))) as Session;
    return normalizeSession(parsed);
  } catch {
    return null;
  }
}

export async function writeSession(userId: string, session: Session): Promise<void> {
  await writeData(getSessionPath(userId), JSON.stringify(normalizeSession(session), null, 2));
}

/** Read the research question from ``data/{userId}/query.txt``. */
export async function readQuestion(userId: string): Promise<string | undefined> {
  try {
    const text = await readData(`${userId}/query.txt`);
    return text.trim() || undefined;
  } catch {
    return undefined;
  }
}
