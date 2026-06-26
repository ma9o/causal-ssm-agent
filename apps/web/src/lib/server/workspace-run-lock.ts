import { randomUUID } from "node:crypto";
import { createControlStoreClient } from "@/lib/server/control-store";

const WORKSPACE_RUN_SLOT_TABLE = "workspace_run_slots";
const WORKSPACE_RUN_RESERVATION_TTL_MS = 15 * 60 * 1000;
const CLAIM_MAX_ATTEMPTS = 5;

type WorkspaceRunSlotRow = {
  reservationId: string;
  reservedAtMs: number;
};

export type WorkspaceRunSlotClaim =
  | { status: "claimed"; reservationId: string }
  | { status: "busy" };

async function ensureSchema(): Promise<void> {
  const client = createControlStoreClient();
  await client.execute(`
    CREATE TABLE IF NOT EXISTS ${WORKSPACE_RUN_SLOT_TABLE} (
      workspace_id TEXT PRIMARY KEY,
      reservation_id TEXT NOT NULL,
      reserved_at_ms INTEGER NOT NULL
    )
  `);
}

function coerceRow(value: unknown): WorkspaceRunSlotRow | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const row = value as Record<string, unknown>;
  const reservationId = row.reservation_id;
  const reservedAtMs = row.reserved_at_ms;

  if (typeof reservationId !== "string") {
    return null;
  }

  const reservedAt =
    typeof reservedAtMs === "number"
      ? reservedAtMs
      : typeof reservedAtMs === "bigint"
        ? Number(reservedAtMs)
        : typeof reservedAtMs === "string"
          ? Number.parseInt(reservedAtMs, 10)
          : Number.NaN;

  if (!Number.isFinite(reservedAt)) {
    return null;
  }

  return {
    reservationId,
    reservedAtMs: reservedAt,
  };
}

async function getSlot(workspaceId: string): Promise<WorkspaceRunSlotRow | null> {
  const client = createControlStoreClient();
  const result = await client.execute({
    sql: `SELECT reservation_id, reserved_at_ms
          FROM ${WORKSPACE_RUN_SLOT_TABLE}
          WHERE workspace_id = ?`,
    args: [workspaceId],
  });

  return coerceRow(result.rows[0]);
}

async function tryInsertReservation(
  workspaceId: string,
  reservationId: string,
  nowMs: number,
): Promise<boolean> {
  const client = createControlStoreClient();
  const result = await client.execute({
    sql: `INSERT OR IGNORE INTO ${WORKSPACE_RUN_SLOT_TABLE}
          (workspace_id, reservation_id, reserved_at_ms)
          VALUES (?, ?, ?)
          RETURNING workspace_id`,
    args: [workspaceId, reservationId, nowMs],
  });

  return result.rows.length > 0;
}

async function deletePendingReservation(
  workspaceId: string,
  reservationId: string,
  reservedAtMs: number,
): Promise<boolean> {
  const client = createControlStoreClient();
  const result = await client.execute({
    sql: `DELETE FROM ${WORKSPACE_RUN_SLOT_TABLE}
          WHERE workspace_id = ?
            AND reservation_id = ?
            AND reserved_at_ms = ?
          RETURNING workspace_id`,
    args: [workspaceId, reservationId, reservedAtMs],
  });

  return result.rows.length > 0;
}

export async function claimWorkspaceRunSlot(workspaceId: string): Promise<WorkspaceRunSlotClaim> {
  await ensureSchema();

  for (let attempt = 0; attempt < CLAIM_MAX_ATTEMPTS; attempt += 1) {
    const reservationId = randomUUID();
    const nowMs = Date.now();

    if (await tryInsertReservation(workspaceId, reservationId, nowMs)) {
      return { status: "claimed", reservationId };
    }

    const existing = await getSlot(workspaceId);
    if (!existing) {
      continue;
    }

    if (nowMs - existing.reservedAtMs > WORKSPACE_RUN_RESERVATION_TTL_MS) {
      await deletePendingReservation(workspaceId, existing.reservationId, existing.reservedAtMs);
      continue;
    }

    return { status: "busy" };
  }

  throw new Error(`Failed to claim workspace run slot for ${workspaceId}`);
}

export async function releaseWorkspaceRunSlot(
  workspaceId: string,
  reservationId: string,
): Promise<void> {
  const client = createControlStoreClient();
  await client.execute({
    sql: `DELETE FROM ${WORKSPACE_RUN_SLOT_TABLE}
          WHERE workspace_id = ?
            AND reservation_id = ?`,
    args: [workspaceId, reservationId],
  });
}
