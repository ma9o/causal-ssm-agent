import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { readdir, readFile } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const WORKSPACE_ID = "DEMO";

function getRequiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`${name} is required to upload ${WORKSPACE_ID} artifacts to R2.`);
  }
  return value;
}

function normalizeR2Prefix(prefix: string): string {
  const normalized = prefix.replace(/^\/+|\/+$/g, "");
  if (!normalized) {
    throw new Error("R2_PREFIX must not be empty.");
  }
  return normalized;
}

async function listFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        return listFiles(path);
      }
      return entry.isFile() ? [path] : [];
    }),
  );
  return files.flat().sort((left, right) => left.localeCompare(right));
}

function contentTypeFor(path: string): string {
  if (path.endsWith(".json")) return "application/json";
  if (path.endsWith(".csv")) return "text/csv";
  if (path.endsWith(".md")) return "text/markdown";
  if (path.endsWith(".txt")) return "text/plain";
  if (path.endsWith(".ics")) return "text/calendar";
  if (path.endsWith(".xml")) return "application/xml";
  if (path.endsWith(".zip")) return "application/zip";
  if (path.endsWith(".parquet")) return "application/vnd.apache.parquet";
  if (path.endsWith(".pkl")) return "application/octet-stream";
  return "application/octet-stream";
}

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "..", "..", "..");
const demoDirectory = join(repoRoot, "data", WORKSPACE_ID);

const bucket = getRequiredEnv("R2_BUCKET");
const prefix = normalizeR2Prefix(getRequiredEnv("R2_PREFIX"));
const client = new S3Client({
  credentials: {
    accessKeyId: getRequiredEnv("R2_ACCESS_KEY_ID"),
    secretAccessKey: getRequiredEnv("R2_SECRET_ACCESS_KEY"),
  },
  endpoint: getRequiredEnv("R2_ENDPOINT_URL"),
  region: "auto",
});

const files = await listFiles(demoDirectory);
if (files.length === 0) {
  throw new Error(`${demoDirectory} does not contain any files to upload.`);
}

for (const file of files) {
  const localPath = relative(demoDirectory, file).split(sep).join("/");
  const key = `${prefix}/${WORKSPACE_ID}/${localPath}`;
  const body = await readFile(file);
  await client.send(
    new PutObjectCommand({
      Body: body,
      Bucket: bucket,
      ContentLength: body.byteLength,
      ContentType: contentTypeFor(file),
      Key: key,
    }),
  );
}

console.log(
  `Uploaded ${files.length} ${WORKSPACE_ID} artifact(s) to R2 prefix ${prefix}/${WORKSPACE_ID}/.`,
);
