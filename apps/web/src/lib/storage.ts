/**
 * Pluggable storage backend — local filesystem or Cloudflare R2.
 *
 * Production (`DEPLOYMENT_ENV=production`) uses Cloudflare R2.
 * All other environments default to local filesystem.
 *
 * Environment variables for R2:
 *   DEPLOYMENT_ENV=production
 *   R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
 *   R2_ACCESS_KEY_ID=...
 *   R2_SECRET_ACCESS_KEY=...
 *   R2_BUCKET=...
 *   R2_PREFIX=data              // key prefix inside bucket (default: "data")
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

const isRemote = process.env.DEPLOYMENT_ENV === "production";

// ---------------------------------------------------------------------------
// S3 client (lazy-initialized for R2)
// ---------------------------------------------------------------------------

let _s3: import("@aws-sdk/client-s3").S3Client | null = null;

function getS3(): import("@aws-sdk/client-s3").S3Client {
  if (_s3) return _s3;
  // Dynamic import to avoid loading the SDK in local mode
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { S3Client } = require("@aws-sdk/client-s3") as typeof import("@aws-sdk/client-s3");
  _s3 = new S3Client({
    region: "auto",
    endpoint: process.env.R2_ENDPOINT_URL!,
    credentials: {
      accessKeyId: process.env.R2_ACCESS_KEY_ID!,
      secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
    },
  });
  return _s3;
}

const BUCKET = process.env.R2_BUCKET ?? "";
const PREFIX = process.env.R2_PREFIX ?? "data";

/**
 * Local DATA_DIR — only used in local mode for path safety checks.
 */
export const LOCAL_DATA_DIR = resolve(process.cwd(), "..", "..", "data");

function r2Key(relativePath: string): string {
  return `${PREFIX}/${relativePath}`;
}

// ---------------------------------------------------------------------------
// Read / write helpers
// ---------------------------------------------------------------------------

/**
 * Read a text file from storage.
 * @param relativePath  Path relative to the data root (e.g. "userId/run/stage-1a.json")
 */
export async function readData(relativePath: string): Promise<string> {
  if (isRemote) {
    const { GetObjectCommand } = await import("@aws-sdk/client-s3");
    const resp = await getS3().send(
      new GetObjectCommand({ Bucket: BUCKET, Key: r2Key(relativePath) }),
    );
    return (await resp.Body!.transformToString("utf-8"));
  }
  return readFile(resolve(LOCAL_DATA_DIR, relativePath), "utf-8");
}

/**
 * Write a text file to storage.
 */
export async function writeData(relativePath: string, content: string): Promise<void> {
  if (isRemote) {
    const { PutObjectCommand } = await import("@aws-sdk/client-s3");
    await getS3().send(
      new PutObjectCommand({
        Bucket: BUCKET,
        Key: r2Key(relativePath),
        Body: content,
        ContentType: "application/octet-stream",
      }),
    );
    return;
  }
  const fullPath = resolve(LOCAL_DATA_DIR, relativePath);
  await mkdir(dirname(fullPath), { recursive: true });
  await writeFile(fullPath, content, "utf-8");
}

/**
 * Write binary data to storage.
 */
export async function writeBinary(relativePath: string, data: Buffer): Promise<void> {
  if (isRemote) {
    const { PutObjectCommand } = await import("@aws-sdk/client-s3");
    await getS3().send(
      new PutObjectCommand({
        Bucket: BUCKET,
        Key: r2Key(relativePath),
        Body: data,
        ContentType: "application/octet-stream",
      }),
    );
    return;
  }
  const fullPath = resolve(LOCAL_DATA_DIR, relativePath);
  await mkdir(dirname(fullPath), { recursive: true });
  await writeFile(fullPath, data);
}

/**
 * Ensure a directory exists. No-op for R2 (directories are implicit).
 */
export async function ensureDir(relativePath: string): Promise<void> {
  if (isRemote) return;
  await mkdir(resolve(LOCAL_DATA_DIR, relativePath), { recursive: true });
}

/**
 * Resolve a relative data path to a local absolute path.
 * Only valid in local mode — used for path safety checks.
 */
export function localResolve(relativePath: string): string {
  return resolve(join(LOCAL_DATA_DIR, relativePath));
}
