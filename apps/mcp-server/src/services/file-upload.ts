import { copyFile, mkdir, stat } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const DATA_DIR =
  process.env.DATA_DIR ??
  resolve(__dirname, "..", "..", "..", "data-pipeline", "data", "raw");

export async function uploadDataFile(
  sourcePath: string,
  userId: string,
): Promise<string> {
  const info = await stat(sourcePath).catch(() => null);
  if (!info?.isFile()) {
    throw new Error(`File not found: ${sourcePath}`);
  }

  const filename = basename(sourcePath);
  if (!filename.endsWith(".zip")) {
    throw new Error(`Expected a .zip file, got: ${filename}`);
  }

  const destDir = resolve(DATA_DIR, basename(userId));
  await mkdir(destDir, { recursive: true });

  const destPath = resolve(destDir, filename);
  await copyFile(sourcePath, destPath);

  return destPath;
}
