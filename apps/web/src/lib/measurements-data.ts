import type {
  ObservationRecord,
  MeasurementsData,
  MeasurementsPersistedData,
} from "@nof1-causal-lab/api-types";
import {
  normalizeParquetScalar,
  type ParquetMetadata,
  readParquetRows,
  toArrayBuffer,
} from "./parquet-utils";

function normalizeOptionalString(value: unknown): string | null | undefined {
  if (value == null) return null;
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "string") return value;
  if (typeof value === "bigint" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return undefined;
}

function normalizeObservationRecord(row: Record<string, unknown>): ObservationRecord {
  return {
    indicator: String(row.indicator ?? ""),
    value: normalizeParquetScalar(row.value),
    anchor_time: normalizeOptionalString(row.anchor_time) ?? null,
    support_kind: normalizeOptionalString(row.support_kind),
    summary_operator: normalizeOptionalString(row.summary_operator),
    anchor_policy: normalizeOptionalString(row.anchor_policy),
    observation_window: normalizeOptionalString(row.observation_window),
    support_start: normalizeOptionalString(row.support_start),
    support_end: normalizeOptionalString(row.support_end),
  };
}

async function derivePerIndicatorCounts(
  file: ArrayBuffer,
  metadata: ParquetMetadata,
): Promise<Record<string, number>> {
  const totalRows = Number(metadata.num_rows);
  if (totalRows === 0) return {};

  const rows = await readParquetRows(file, metadata, 0, totalRows, ["indicator"]);
  const counts = new Map<string, number>();

  for (const row of rows) {
    const indicator = row.indicator;
    if (typeof indicator !== "string" || indicator.length === 0) continue;
    counts.set(indicator, (counts.get(indicator) ?? 0) + 1);
  }

  return Object.fromEntries(
    [...counts.entries()].sort(([left], [right]) => left.localeCompare(right)),
  );
}

async function deriveExtractionSample(
  file: ArrayBuffer,
  metadata: ParquetMetadata,
): Promise<ObservationRecord[]> {
  const sampleSize = Math.min(Number(metadata.num_rows), 20);
  if (sampleSize === 0) return [];

  const rows = await readParquetRows(file, metadata, 0, sampleSize);
  return rows.map(normalizeObservationRecord);
}

export async function deriveMeasurementsData(
  payload: MeasurementsPersistedData,
  parquetBytes: Uint8Array,
): Promise<MeasurementsData> {
  const file = toArrayBuffer(parquetBytes);
  const { parquetMetadata } = await import("hyparquet");
  const metadata = parquetMetadata(file) as ParquetMetadata;

  return {
    workers: payload.workers ?? [],
    per_indicator_counts: await derivePerIndicatorCounts(file, metadata),
    combined_extractions_sample: await deriveExtractionSample(file, metadata),
  };
}
