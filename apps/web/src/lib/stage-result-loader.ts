import type { Stage0PersistedData, Stage2PersistedData, StageId } from "@causal-ssm/api-types";
import { deriveStage0Data } from "@/lib/stage0-data";
import { deriveStage2Data } from "@/lib/stage2-data";
import { readBinary } from "@/lib/storage";

type StageResultLoader = (payload: unknown, workspaceId: string) => Promise<unknown>;

function normalizeNonFiniteJsonTokens(serialized: string): string {
  let normalized = "";
  let inString = false;
  let escaping = false;

  for (let index = 0; index < serialized.length; index += 1) {
    const char = serialized[index];

    if (inString) {
      normalized += char;
      if (escaping) {
        escaping = false;
      } else if (char === "\\") {
        escaping = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }

    if (char === "\"") {
      inString = true;
      normalized += char;
      continue;
    }

    if (serialized.startsWith("-Infinity", index)) {
      normalized += "null";
      index += "-Infinity".length - 1;
      continue;
    }

    if (serialized.startsWith("Infinity", index)) {
      normalized += "null";
      index += "Infinity".length - 1;
      continue;
    }

    if (serialized.startsWith("NaN", index)) {
      normalized += "null";
      index += "NaN".length - 1;
      continue;
    }

    normalized += char;
  }

  return normalized;
}

function parseStoredStagePayload(raw: string): unknown {
  const parsed = JSON.parse(raw);

  if (!parsed?.metadata || typeof parsed.result !== "string") {
    return parsed;
  }

  try {
    return JSON.parse(parsed.result);
  } catch {
    return JSON.parse(normalizeNonFiniteJsonTokens(parsed.result));
  }
}

const STAGE_RESULT_LOADERS: Partial<Record<StageId, StageResultLoader>> = {
  "stage-0": async (payload, workspaceId) => {
    const parquet = await readBinary(`${workspaceId}/run/stage0-raw-input.parquet`);
    return deriveStage0Data(payload as Stage0PersistedData, parquet);
  },
  "stage-2": async (payload, workspaceId) => {
    const parquet = await readBinary(`${workspaceId}/run/stage2-model-data.parquet`);
    return deriveStage2Data(payload as Stage2PersistedData, parquet);
  },
};

export async function loadStageResult(stageId: string, raw: string, workspaceId: string): Promise<unknown> {
  const payload = parseStoredStagePayload(raw);
  const loader = STAGE_RESULT_LOADERS[stageId as StageId];
  return loader ? loader(payload, workspaceId) : payload;
}
