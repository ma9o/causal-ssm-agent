const CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"; // 31 chars, no 0/1/I/O
export const ANONYMOUS_WORKSPACE_ID_LENGTH = 12;

const MAX_WORKSPACE_ID_LENGTH = 200;

export function generateAnonymousWorkspaceId(): string {
  const values = crypto.getRandomValues(new Uint8Array(ANONYMOUS_WORKSPACE_ID_LENGTH));
  return Array.from(values, (v) => CHARSET[v % CHARSET.length]).join("");
}

/** Trim and validate a workspace id for use in storage paths; null when malformed. */
export function normalizeWorkspaceId(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > MAX_WORKSPACE_ID_LENGTH) {
    return null;
  }

  if (!/^[A-Za-z0-9_-]+$/.test(trimmed)) {
    return null;
  }

  return trimmed;
}
