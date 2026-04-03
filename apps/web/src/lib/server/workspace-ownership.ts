import { readdir } from "node:fs/promises";
import { SHARED_WORKSPACE_IDS } from "@/lib/shared-workspaces";
import {
  LOCAL_DATA_DIR,
  isStorageNotFoundError,
  prefixExists,
  readData,
} from "@/lib/storage";
import { createControlStoreClient } from "@/lib/server/control-store";
import { resolveOpenRouterAccess } from "@/lib/server/openrouter-access";
import { readAuthorizedWorkspaceIds } from "@/lib/server/workspace-session";

const MAX_PERSISTED_WORKSPACES = 256;
const WORKSPACE_OWNERSHIP_TABLE = "workspace_ownership";

type WorkspaceOwnershipRow = {
  ownerUserId: string;
  updatedAtMs: number;
  workspaceId: string;
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

async function ensureWorkspaceOwnershipSchema(): Promise<void> {
  const client = createControlStoreClient();

  await client.batch(
    [
      {
        sql: `CREATE TABLE IF NOT EXISTS ${WORKSPACE_OWNERSHIP_TABLE} (
                workspace_id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL
              )`,
      },
      {
        sql: `CREATE INDEX IF NOT EXISTS ${WORKSPACE_OWNERSHIP_TABLE}_owner_idx
              ON ${WORKSPACE_OWNERSHIP_TABLE} (owner_user_id, updated_at_ms DESC)`,
      },
    ],
    "write",
  );
}

function coerceOwnershipRow(value: unknown): WorkspaceOwnershipRow | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const row = value as Record<string, unknown>;
  const workspaceId = row.workspace_id;
  const ownerUserId = row.owner_user_id;
  const updatedAtMs = row.updated_at_ms;

  if (typeof workspaceId !== "string" || typeof ownerUserId !== "string") {
    return null;
  }

  const coerceNumber = (input: unknown): number => {
    if (typeof input === "number") {
      return input;
    }
    if (typeof input === "bigint") {
      return Number(input);
    }
    if (typeof input === "string") {
      return Number.parseInt(input, 10);
    }
    return Number.NaN;
  };

  const updated = coerceNumber(updatedAtMs);
  if (!Number.isFinite(updated)) {
    return null;
  }

  return {
    ownerUserId,
    updatedAtMs: updated,
    workspaceId,
  };
}

async function listOpenRouterWorkspaceRows(userId: string): Promise<WorkspaceOwnershipRow[]> {
  await ensureWorkspaceOwnershipSchema();

  const client = createControlStoreClient();
  const result = await client.execute({
    sql: `SELECT workspace_id, owner_user_id, updated_at_ms
          FROM ${WORKSPACE_OWNERSHIP_TABLE}
          WHERE owner_user_id = ?
          ORDER BY updated_at_ms DESC
          LIMIT ?`,
    args: [userId, MAX_PERSISTED_WORKSPACES],
  });

  return result.rows
    .map(coerceOwnershipRow)
    .filter((row): row is WorkspaceOwnershipRow => row !== null);
}

async function persistOpenRouterWorkspaceOwnership(
  userId: string,
  workspaceId: string,
  updatedAtMs: number,
): Promise<void> {
  await ensureWorkspaceOwnershipSchema();

  const client = createControlStoreClient();
  const result = await client.execute({
    sql: `INSERT INTO ${WORKSPACE_OWNERSHIP_TABLE}
            (workspace_id, owner_user_id, updated_at_ms)
          VALUES (?, ?, ?)
          ON CONFLICT(workspace_id) DO UPDATE SET
            updated_at_ms = excluded.updated_at_ms
          WHERE ${WORKSPACE_OWNERSHIP_TABLE}.owner_user_id = excluded.owner_user_id
          RETURNING workspace_id`,
    args: [workspaceId, userId, updatedAtMs],
  });

  if (result.rows.length === 0) {
    throw new Error(`Workspace '${workspaceId}' is already owned by another user.`);
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
  return (await listOpenRouterWorkspaceRows(userId)).map((row) => row.workspaceId);
}

export async function hasOpenRouterWorkspaceAccess(
  userId: string,
  workspaceId: string,
): Promise<boolean> {
  await ensureWorkspaceOwnershipSchema();

  const client = createControlStoreClient();
  const result = await client.execute({
    sql: `SELECT workspace_id, owner_user_id, updated_at_ms
          FROM ${WORKSPACE_OWNERSHIP_TABLE}
          WHERE workspace_id = ?
            AND owner_user_id = ?
          LIMIT 1`,
    args: [workspaceId, userId],
  });
  return coerceOwnershipRow(result.rows[0]) !== null;
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
  const normalizedWorkspaceIds = deduplicateWorkspaceIds(workspaceIds);
  const baseUpdatedAtMs = Date.now();

  for (const [index, workspaceId] of normalizedWorkspaceIds.entries()) {
    await persistOpenRouterWorkspaceOwnership(
      userId,
      workspaceId,
      baseUpdatedAtMs - index,
    );
  }
}

export async function listLocalWorkspaceIds(): Promise<string[]> {
  try {
    const entries = await readdir(LOCAL_DATA_DIR, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
      .map((entry) => entry.name)
      .sort((left, right) => left.localeCompare(right));
  } catch (e: unknown) {
    if (e instanceof Error && "code" in e && (e as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }
    throw e;
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

async function readWorkspaceQuestion(workspaceId: string): Promise<string | null> {
  try {
    const text = (await readData(`${workspaceId}/query.txt`)).trim();
    if (!text) {
      return null;
    }
    return text.length > 120 ? `${text.slice(0, 117)}...` : text;
  } catch (e: unknown) {
    if (isStorageNotFoundError(e)) {
      return null;
    }
    throw e;
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
        await readOpenRouterOwnedWorkspaceIds(ownership.userId),
        "user",
      ),
    );
  } else {
    const sharedSet = new Set(sharedWorkspaceIds);
    const sessionWorkspaceIds = (
      await filterExistingWorkspaceIds(
        deduplicateWorkspaceIds(await readAuthorizedWorkspaceIds()),
      )
    ).filter((workspaceId) => !sharedSet.has(workspaceId));
    entries.push(...await buildEntries(sessionWorkspaceIds, "session"));
  }

  entries.push(...await buildEntries(sharedWorkspaceIds, "shared"));

  return {
    mode: ownership.mode,
    workspaces: entries,
  };
}
