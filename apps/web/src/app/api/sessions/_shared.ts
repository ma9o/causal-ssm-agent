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

function getSessionPath(workspaceId: string): string {
  return `${workspaceId}/session.json`;
}

export async function readSession(workspaceId: string): Promise<Session | null> {
  try {
    const parsed = JSON.parse(await readData(getSessionPath(workspaceId))) as Session;
    return normalizeSession(parsed);
  } catch {
    return null;
  }
}

export async function writeSession(workspaceId: string, session: Session): Promise<void> {
  await writeData(getSessionPath(workspaceId), JSON.stringify(normalizeSession(session), null, 2));
}

/** Read the research question from ``data/{workspaceId}/query.txt``. */
export async function readQuestion(workspaceId: string): Promise<string | undefined> {
  try {
    const text = await readData(`${workspaceId}/query.txt`);
    return text.trim() || undefined;
  } catch {
    return undefined;
  }
}
