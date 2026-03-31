const CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"; // 31 chars, no 0/1/I/O
export const ANONYMOUS_WORKSPACE_ID_LENGTH = 12;

export function generateAnonymousWorkspaceId(): string {
  const values = crypto.getRandomValues(new Uint8Array(ANONYMOUS_WORKSPACE_ID_LENGTH));
  return Array.from(values, (v) => CHARSET[v % CHARSET.length]).join("");
}
