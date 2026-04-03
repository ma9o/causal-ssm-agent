import { readdir } from "node:fs/promises";
import { SHARED_WORKSPACE_IDS } from "@/lib/shared-workspaces";
import { LOCAL_DATA_DIR, prefixExists, readData, writeData } from "@/lib/storage";
import { resolveOpenRouterAccess } from "@/lib/server/openrouter-access";
import { readAuthorizedWorkspaceIds } from "@/lib/server/workspace-session";

const MAX_PERSISTED_WORKSPACES = 256;

type OpenRouterWorkspaceIndex = {
  workspaceIds?: string[];
};

export type WorkspaceOwnershipContext =
  | { mode: "anonymous" }
  | { mode: "user"; userId: string }
  | { mode: "local" };

export type AccessibleWorkspaceSource = "user" | "session" | "local" | "shared";

export type AccessibleWorkspaceEntry = {
  href: string;
  question: string | null;
  source: AccessibleWorkspaceSource;
  workspaceId: string;
};

export type AccessibleWorkspaceList = {
  mode: WorkspaceOwnershipContext["mode"];
  workspaces: AccessibleWorkspaceEntry[];
};

function deduplicateWorkspaceIds(workspaceIds: string[]): string[] {
  const normalized = new Set<string>();
  for (const workspaceId of workspaceIds) {
    const trimmed = workspaceId.trim();
    if (!trimmed || trimmed.startsWith(".")) {
      continue;
    }
    normalized.add(trimmed);
  }
  return [...normalized].slice(0, MAX_PERSISTED_WORKSPACES);
}

function openRouterWorkspaceIndexPath(userId: string): string {
  return `.private/openrouter-users/${Buffer.from(userId, "utf-8").toString("base64url")}/workspaces.json`;
}

async function readOpenRouterWorkspaceIndex(userId: string): Promise<OpenRouterWorkspaceIndex> {
  try {
    return JSON.parse(await readData(openRouterWorkspaceIndexPath(userId))) as OpenRouterWorkspaceIndex;
  } catch {
    return {};
  }
}

export async function resolveWorkspaceOwnershipContext(): Promise<WorkspaceOwnershipContext> {
  const access = await resolveOpenRouterAccess();
  switch (access.mode) {
    case "local":
      return { mode: "local" };
    case "user":
      return { mode: "user", userId: access.userId };
    case "anonymous":
      return { mode: "anonymous" };
    case "none":
      return access.reason === "local_missing_key" ? { mode: "local" } : { mode: "anonymous" };
  }
}

export async function readOpenRouterOwnedWorkspaceIds(userId: string): Promise<string[]> {
  const index = await readOpenRouterWorkspaceIndex(userId);
  return index.workspaceIds ?? [];
}

export async function hasOpenRouterWorkspaceAccess(
  userId: string,
  workspaceId: string,
): Promise<boolean> {
  const workspaceIds = await readOpenRouterOwnedWorkspaceIds(userId);
  return workspaceIds.includes(workspaceId);
}

export async function authorizeWorkspaceForOpenRouterUser(
  userId: string,
  workspaceId: string,
): Promise<void> {
  await authorizeWorkspacesForOpenRouterUser(userId, [workspaceId]);
}

export async function authorizeWorkspacesForOpenRouterUser(
  userId: string,
  workspaceIds: string[],
): Promise<void> {
  const existing = await readOpenRouterOwnedWorkspaceIds(userId);
  const merged = deduplicateWorkspaceIds([...workspaceIds, ...existing]);
  await writeData(
    openRouterWorkspaceIndexPath(userId),
    JSON.stringify({ workspaceIds: merged }, null, 2),
  );
}

export async function listLocalWorkspaceIds(): Promise<string[]> {
  try {
    const entries = await readdir(LOCAL_DATA_DIR, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
      .map((entry) => entry.name)
      .sort((left, right) => left.localeCompare(right));
  } catch {
    return [];
  }
}

async function listAvailableSharedWorkspaceIds(): Promise<string[]> {
  const checks = await Promise.all(
    SHARED_WORKSPACE_IDS.map(async (workspaceId) => ({
      exists: await prefixExists(`${workspaceId}/`),
      workspaceId,
    })),
  );

  return checks.filter((entry) => entry.exists).map((entry) => entry.workspaceId);
}

async function filterExistingWorkspaceIds(workspaceIds: string[]): Promise<string[]> {
  const checks = await Promise.all(
    workspaceIds.map(async (workspaceId) => ({
      exists: await prefixExists(`${workspaceId}/`),
      workspaceId,
    })),
  );

  return checks.filter((entry) => entry.exists).map((entry) => entry.workspaceId);
}

async function readAuthorizedWorkspaceIdsSafely(): Promise<string[]> {
  try {
    return deduplicateWorkspaceIds(await readAuthorizedWorkspaceIds());
  } catch {
    return [];
  }
}

async function readWorkspaceQuestion(workspaceId: string): Promise<string | null> {
  try {
    const text = (await readData(`${workspaceId}/query.txt`)).trim();
    if (!text) {
      return null;
    }
    return text.length > 120 ? `${text.slice(0, 117)}...` : text;
  } catch {
    return null;
  }
}

async function buildEntries(
  workspaceIds: string[],
  source: AccessibleWorkspaceSource,
): Promise<AccessibleWorkspaceEntry[]> {
  return Promise.all(
    workspaceIds.map(async (workspaceId) => ({
      href: `/analysis/${workspaceId}`,
      question: await readWorkspaceQuestion(workspaceId),
      source,
      workspaceId,
    })),
  );
}

export async function listAccessibleWorkspaces(): Promise<AccessibleWorkspaceList> {
  const ownership = await resolveWorkspaceOwnershipContext();

  if (ownership.mode === "local") {
    return {
      mode: "local",
      workspaces: await buildEntries(await listLocalWorkspaceIds(), "local"),
    };
  }

  const sharedWorkspaceIds = await listAvailableSharedWorkspaceIds();
  const entries: AccessibleWorkspaceEntry[] = [];

  if (ownership.mode === "user") {
    entries.push(
      ...await buildEntries(
        await filterExistingWorkspaceIds(await readOpenRouterOwnedWorkspaceIds(ownership.userId)),
        "user",
      ),
    );
  } else {
    const sharedSet = new Set(sharedWorkspaceIds);
    const sessionWorkspaceIds = (
      await filterExistingWorkspaceIds(await readAuthorizedWorkspaceIdsSafely())
    ).filter((workspaceId) => !sharedSet.has(workspaceId));
    entries.push(...await buildEntries(sessionWorkspaceIds, "session"));
  }

  entries.push(...await buildEntries(sharedWorkspaceIds, "shared"));

  return {
    mode: ownership.mode,
    workspaces: entries,
  };
}
