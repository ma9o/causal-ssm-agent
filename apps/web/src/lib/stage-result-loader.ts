import type {
  Stage0PersistedData,
  Stage2PersistedData,
  Stage3Data,
  Stage4PersistedData,
  StageId,
} from "@nof1-causal-lab/api-types";
import { deriveStage0Data } from "@/lib/stage0-data";
import { deriveStage2Data } from "@/lib/stage2-data";
import { deriveStage4Data } from "@/lib/stage4-derived-data";
import { readBinary, readData } from "@/lib/storage";

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
  let parsed: unknown;

  try {
    parsed = JSON.parse(raw);
  } catch (parseError) {
    try {
      parsed = JSON.parse(normalizeNonFiniteJsonTokens(raw));
    } catch (normalizeError) {
      throw new Error(
        `Failed to parse stage result JSON: ${normalizeError instanceof Error ? normalizeError.message : String(normalizeError)}`,
      );
    }
  }

  if (!parsed?.metadata || typeof parsed.result !== "string") {
    return parsed;
  }

  try {
    return JSON.parse(parsed.result);
  } catch (parseError) {
    try {
      return JSON.parse(normalizeNonFiniteJsonTokens(parsed.result));
    } catch (normalizeError) {
      throw new Error(
        `Failed to parse stage result JSON: ${normalizeError instanceof Error ? normalizeError.message : String(normalizeError)}`,
      );
    }
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
  "stage-4": async (payload, workspaceId) => {
    const [parquet, stage3Raw] = await Promise.all([
      readBinary(`${workspaceId}/run/stage2-model-data.parquet`),
      readData(`${workspaceId}/run/stage-3.json`),
    ]);
    return deriveStage4Data(
      payload as Stage4PersistedData,
      parseStoredStagePayload(stage3Raw) as Stage3Data,
      parquet,
    );
  },
};

export async function loadStageResult(stageId: string, raw: string, workspaceId: string): Promise<unknown> {
  const payload = parseStoredStagePayload(raw);
  const loader = STAGE_RESULT_LOADERS[stageId as StageId];
  return loader ? loader(payload, workspaceId) : payload;
}
