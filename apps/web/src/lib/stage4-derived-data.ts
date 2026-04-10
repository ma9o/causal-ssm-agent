import type {
  ObservationRecord,
  Stage3Data,
  Stage4Data,
  Stage4PersistedData,
} from "@causal-ssm/api-types";
import type { FileMetaData } from "hyparquet";
import { buildStage4LikelihoodDiagnostics } from "./stage4-likelihood-diagnostics";

type ParquetSchemaColumn = {
  name: string;
  num_children?: number;
};

type ParquetMetadata = {
  num_rows: bigint | number;
  schema: ParquetSchemaColumn[];
};

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

async function readRows(
  file: ArrayBuffer,
  metadata: ParquetMetadata,
  rowStart: number,
  rowEnd: number,
  columns?: string[],
): Promise<Record<string, unknown>[]> {
  const { parquetReadObjects } = await import("hyparquet");
  const { compressors } = await import("hyparquet-compressors");

  return parquetReadObjects({
    file,
    metadata: metadata as unknown as FileMetaData,
    compressors,
    rowStart,
    rowEnd,
    columns,
  });
}

function normalizeScalar(value: unknown): number | boolean | string | null {
  if (value == null) return null;
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "string") {
    return value;
  }
  if (typeof value === "bigint") {
    return value <= BigInt(Number.MAX_SAFE_INTEGER) && value >= BigInt(Number.MIN_SAFE_INTEGER)
      ? Number(value)
      : value.toString();
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return String(value);
}

async function readObservationRecords(parquetBytes: Uint8Array): Promise<ObservationRecord[]> {
  const file = toArrayBuffer(parquetBytes);
  const { parquetMetadata } = await import("hyparquet");
  const metadata = parquetMetadata(file) as ParquetMetadata;
  const totalRows = Number(metadata.num_rows);

  if (totalRows === 0) {
    return [];
  }

  const rows = await readRows(file, metadata, 0, totalRows, ["indicator", "value"]);
  return rows.map((row) => ({
    indicator: String(row.indicator ?? ""),
    value: normalizeScalar(row.value),
    anchor_time: null,
  }));
}

export async function deriveStage4Data(
  payload: Stage4PersistedData,
  stage3: Stage3Data,
  parquetBytes: Uint8Array,
): Promise<Stage4Data> {
  const observations = await readObservationRecords(parquetBytes);

  return {
    ...payload,
    likelihood_diagnostics: buildStage4LikelihoodDiagnostics({
      likelihoods: payload.model_spec.likelihoods,
      indicatorAudits: stage3.indicators,
      observations,
    }),
  };
}
