const CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"; // 31 chars, no 0/1/I/O
export const ANONYMOUS_USER_ID_LENGTH = 6;

export function generateAnonymousUserId(): string {
  const values = crypto.getRandomValues(new Uint8Array(ANONYMOUS_USER_ID_LENGTH));
  return Array.from(values, (v) => CHARSET[v % CHARSET.length]).join("");
}
