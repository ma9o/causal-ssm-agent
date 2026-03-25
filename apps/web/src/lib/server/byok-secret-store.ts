import { createCipheriv, createDecipheriv, createHash, randomBytes } from "node:crypto";
import type { Client } from "@libsql/client";
import { deriveAppSecret } from "@/lib/server/app-secret";
import { createControlStoreClient } from "@/lib/server/control-store";

const BYOK_SECRET_TABLE = "byok_secret_refs";
const DEFAULT_BYOK_SECRET_TTL_SECONDS = 60 * 60;

function getByokSecretStoreTtlSeconds(): number {
  const rawValue = process.env.BYOK_SECRET_STORE_TTL_SECONDS;
  const parsed = rawValue ? Number.parseInt(rawValue, 10) : Number.NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_BYOK_SECRET_TTL_SECONDS;
}

function getByokSecretStoreSecret(): string {
  try {
    return deriveAppSecret("byok-secret-store");
  } catch {
    throw new Error("APP_SECRET is not configured");
  }
}

function getCipherKey(): Buffer {
  return createHash("sha256").update(getByokSecretStoreSecret(), "utf8").digest();
}

function createByokSecretClient(): Client {
  return createControlStoreClient();
}

async function ensureSchema(client: Client): Promise<void> {
  await client.execute(`
    CREATE TABLE IF NOT EXISTS ${BYOK_SECRET_TABLE} (
      ref TEXT PRIMARY KEY,
      ciphertext TEXT NOT NULL,
      created_at_ms INTEGER NOT NULL,
      expires_at_ms INTEGER NOT NULL
    )
  `);
}

function encodeBase64Url(buffer: Buffer): string {
  return buffer.toString("base64url");
}

function decodeBase64Url(value: string): Buffer {
  return Buffer.from(value, "base64url");
}

function encryptApiKey(apiKey: string): string {
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", getCipherKey(), iv);
  const ciphertext = Buffer.concat([cipher.update(apiKey, "utf8"), cipher.final()]);
  const authTag = cipher.getAuthTag();
  return `v1.${encodeBase64Url(iv)}.${encodeBase64Url(ciphertext)}.${encodeBase64Url(authTag)}`;
}

export function decryptByokSecretPayload(payload: string): string {
  const [version, ivPart, ciphertextPart, authTagPart] = payload.split(".");
  if (version !== "v1" || !ivPart || !ciphertextPart || !authTagPart) {
    throw new Error("Invalid BYOK secret payload");
  }

  const decipher = createDecipheriv("aes-256-gcm", getCipherKey(), decodeBase64Url(ivPart));
  decipher.setAuthTag(decodeBase64Url(authTagPart));
  const plaintext = Buffer.concat([
    decipher.update(decodeBase64Url(ciphertextPart)),
    decipher.final(),
  ]);
  return plaintext.toString("utf8");
}

export async function createByokSecretRef(apiKey: string): Promise<string> {
  const now = Date.now();
  const expiresAtMs = now + getByokSecretStoreTtlSeconds() * 1000;
  const ref = randomBytes(24).toString("base64url");
  const client = createByokSecretClient();

  await ensureSchema(client);
  await client.batch(
    [
      {
        sql: `INSERT INTO ${BYOK_SECRET_TABLE} (ref, ciphertext, created_at_ms, expires_at_ms)
              VALUES (?, ?, ?, ?)`,
        args: [ref, encryptApiKey(apiKey), now, expiresAtMs],
      },
      {
        sql: `DELETE FROM ${BYOK_SECRET_TABLE} WHERE expires_at_ms <= ?`,
        args: [now],
      },
    ],
    "write",
  );

  return ref;
}

export async function deleteByokSecretRef(ref: string): Promise<void> {
  const client = createByokSecretClient();
  await ensureSchema(client);
  await client.execute({
    sql: `DELETE FROM ${BYOK_SECRET_TABLE} WHERE ref = ?`,
    args: [ref],
  });
}
