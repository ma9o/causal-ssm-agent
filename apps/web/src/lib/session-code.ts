const CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"; // 31 chars, no 0/1/I/O
const CODE_LENGTH = 6;

export function generateSessionCode(): string {
  const values = crypto.getRandomValues(new Uint8Array(CODE_LENGTH));
  return Array.from(values, (v) => CHARSET[v % CHARSET.length]).join("");
}
