import { createHmac } from "node:crypto";
import "@/lib/server/root-env";

function getConfiguredAppSecret(): string | undefined {
  const secret = process.env.APP_SECRET?.trim();
  if (!secret) {
    return undefined;
  }
  if (secret.length < 32) {
    throw new Error("APP_SECRET must be set and at least 32 characters");
  }
  return secret;
}

export function getAppSecret(): string {
  const configured = getConfiguredAppSecret();
  if (configured) {
    return configured;
  }
  throw new Error("APP_SECRET is not configured");
}

export function deriveAppSecret(scope: string): string {
  return createHmac("sha256", getAppSecret()).update(scope).digest("hex");
}
