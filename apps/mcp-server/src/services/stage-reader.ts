import { readFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  LARGE_ARRAY_FIELDS,
  LARGE_NESTED_FIELDS,
  type StageId,
} from "../generated/stage-config";

const __dirname = dirname(fileURLToPath(import.meta.url));

const RESULTS_DIR =
  process.env.RESULTS_DIR ??
  resolve(__dirname, "..", "..", "..", "data-pipeline", "results");

export async function readStageResult(
  runId: string,
  stage: StageId,
  includeLargeArrays = false,
): Promise<Record<string, unknown> | null> {
  const safeRunId = basename(runId);
  const safeStage = basename(stage);
  const filePath = resolve(join(RESULTS_DIR, safeRunId, `${safeStage}.json`));

  if (!filePath.startsWith(RESULTS_DIR)) return null;

  try {
    const raw = await readFile(filePath, "utf-8");
    const data = JSON.parse(raw) as Record<string, unknown>;
    if (includeLargeArrays) return data;
    return stripLargeArrays(stage, data);
  } catch {
    return null;
  }
}

function stripLargeArrays(
  stage: StageId,
  data: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...data };

  const topFields = LARGE_ARRAY_FIELDS[stage];
  if (topFields) {
    for (const field of topFields) {
      if (field in result) {
        const val = result[field];
        const desc = Array.isArray(val) ? `${val.length} items` : "large data";
        result[field] = `[omitted — ${desc}]`;
      }
    }
  }

  const nestedSpec = LARGE_NESTED_FIELDS[stage];
  if (nestedSpec) {
    for (const [arrayField, fieldsToStrip] of Object.entries(nestedSpec)) {
      const arr = result[arrayField];
      if (Array.isArray(arr)) {
        result[arrayField] = arr.map((item: unknown) => {
          if (typeof item !== "object" || item === null) return item;
          const cleaned = { ...(item as Record<string, unknown>) };
          for (const f of fieldsToStrip) {
            if (f in cleaned) {
              const val = cleaned[f];
              const desc = Array.isArray(val) ? `${val.length} draws` : "large data";
              cleaned[f] = `[omitted — ${desc}]`;
            }
          }
          return cleaned;
        });
      }
    }
  }

  return result;
}
