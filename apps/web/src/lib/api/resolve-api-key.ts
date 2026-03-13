import { readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Resolve the default (server-side trial) API key from env or data-pipeline .env.
 */
export function getDefaultApiKey(): string | undefined {
  if (process.env.OPENROUTER_API_KEY) return process.env.OPENROUTER_API_KEY;

  // Fallback: read from data-pipeline's .env
  try {
    const envPath = join(process.cwd(), "..", "data-pipeline", ".env");
    const envContent = readFileSync(envPath, "utf-8");
    const match = envContent.match(/^OPENROUTER_API_KEY=(.+)$/m);
    return match?.[1]?.trim();
  } catch {
    return undefined;
  }
}

/**
 * Resolve API key for a request: user key from header > server trial key > 402.
 * Returns `{ key }` or `{ error, status }`.
 */
export function resolveApiKey(req: Request):
  | { key: string; error?: never }
  | { key?: never; error: string; status: number } {
  const userKey = req.headers.get("x-openrouter-key");
  if (userKey) return { key: userKey };

  const serverKey = getDefaultApiKey();
  if (serverKey) return { key: serverKey };

  return { error: "No API key available", status: 402 };
}
