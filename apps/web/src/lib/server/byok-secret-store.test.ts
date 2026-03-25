import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  createByokSecretRef,
  decryptByokSecretPayload,
  deleteByokSecretRef,
} from "./byok-secret-store";

const originalStoreUrl = process.env.BYOK_SECRET_STORE_URL;
const originalAuthToken = process.env.BYOK_SECRET_STORE_AUTH_TOKEN;
const originalAppSecret = process.env.APP_SECRET;

function restoreEnv() {
  if (originalStoreUrl === undefined) {
    delete process.env.BYOK_SECRET_STORE_URL;
  } else {
    process.env.BYOK_SECRET_STORE_URL = originalStoreUrl;
  }

  if (originalAuthToken === undefined) {
    delete process.env.BYOK_SECRET_STORE_AUTH_TOKEN;
  } else {
    process.env.BYOK_SECRET_STORE_AUTH_TOKEN = originalAuthToken;
  }

  if (originalAppSecret === undefined) {
    delete process.env.APP_SECRET;
  } else {
    process.env.APP_SECRET = originalAppSecret;
  }
}

describe("BYOK secret store", () => {
  let tempDir: string | null = null;

  afterEach(() => {
    restoreEnv();
    if (tempDir) {
      rmSync(tempDir, { recursive: true, force: true });
      tempDir = null;
    }
  });

  it("stores encrypted payloads and deletes them by ref", async () => {
    tempDir = mkdtempSync(join(tmpdir(), "byok-secret-store-"));
    const dbPath = join(tempDir, "store.db");
    process.env.BYOK_SECRET_STORE_URL = `file:${dbPath}`;
    process.env.APP_SECRET = "0123456789abcdef0123456789abcdef";
    delete process.env.BYOK_SECRET_STORE_AUTH_TOKEN;

    const { createClient } = await import("@libsql/client");
    const ref = await createByokSecretRef("user-key");
    const db = createClient({ url: `file:${dbPath}` });
    const stored = await db.execute({
      sql: "SELECT ciphertext FROM byok_secret_refs WHERE ref = ?",
      args: [ref],
    });
    const ciphertext = stored.rows[0]?.ciphertext;

    expect(ciphertext).toBeTypeOf("string");
    expect(String(ciphertext).split(".")).toHaveLength(4);
    expect(decryptByokSecretPayload(String(ciphertext))).toBe("user-key");

    await deleteByokSecretRef(ref);

    const deleted = await db.execute({
      sql: "SELECT ciphertext FROM byok_secret_refs WHERE ref = ?",
      args: [ref],
    });
    expect(deleted.rows).toHaveLength(0);
  });
});
