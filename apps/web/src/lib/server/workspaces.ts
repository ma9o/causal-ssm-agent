import { isStorageNotFoundError, listTopLevelDirs, readData } from "@/lib/storage";

export type WorkspaceEntry = {
  href: string;
  question: string | null;
  workspaceId: string;
};

export type WorkspaceList = {
  workspaces: WorkspaceEntry[];
};

/** Current question text from the workspace's episode store, if any. */
async function readWorkspaceQuestion(workspaceId: string): Promise<string | null> {
  try {
    const state = JSON.parse(await readData(`${workspaceId}/episode/state.json`)) as {
      current?: { question?: { version?: number } };
    };
    const version = state.current?.question?.version;
    if (version == null) {
      return null;
    }
    const raw = JSON.parse(
      await readData(`${workspaceId}/store/question/v${version}/question.json`),
    ) as { text?: unknown };
    return typeof raw.text === "string" && raw.text.trim() ? raw.text.trim() : null;
  } catch (e: unknown) {
    if (isStorageNotFoundError(e)) {
      return null;
    }
    throw e;
  }
}

/**
 * Workspaces are whatever lives under the data root: with hosted-interactive
 * mode gone there is no ownership model — a local store lists local work, and
 * the hosted store lists exactly what was deliberately published.
 */
export async function listWorkspaces(): Promise<WorkspaceList> {
  const workspaceIds = await listTopLevelDirs();
  const workspaces = await Promise.all(
    workspaceIds.map(async (workspaceId) => ({
      href: `/analysis/${workspaceId}`,
      question: await readWorkspaceQuestion(workspaceId),
      workspaceId,
    })),
  );
  return { workspaces };
}
