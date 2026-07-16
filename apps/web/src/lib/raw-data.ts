import type { RawDataData, RawDataPersistedData } from "@nof1-causal-lab/api-types";
import type { FileMetaData } from "hyparquet";

type TemporalKind = "timestamp" | "date" | "time";

interface TemporalInfo {
  kind: TemporalKind;
  unit?: string;
}

type ParquetSchemaColumn = {
  name: string;
  type?: string;
  converted_type?: string;
  logical_type?: {
    type?: string;
    unit?: string;
    isAdjustedToUTC?: boolean;
  };
  num_children?: number;
};

type ParquetMetadata = {
  version: number;
  num_rows: bigint | number;
  row_groups: unknown[];
  schema: ParquetSchemaColumn[];
  metadata_length: number;
};

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

function leafColumns(metadata: ParquetMetadata): ParquetSchemaColumn[] {
  return metadata.schema.filter((column, index) => index > 0 && column.num_children == null);
}

function temporalInfo(column: ParquetSchemaColumn): TemporalInfo | null {
  const logicalType = column.logical_type?.type;
  const convertedType = column.converted_type;

  if (
    logicalType === "TIMESTAMP" ||
    convertedType === "TIMESTAMP_MILLIS" ||
    convertedType === "TIMESTAMP_MICROS"
  ) {
    return { kind: "timestamp", unit: column.logical_type?.unit ?? convertedType };
  }
  if (logicalType === "DATE" || convertedType === "DATE") {
    return { kind: "date" };
  }
  if (
    logicalType === "TIME" ||
    convertedType === "TIME_MILLIS" ||
    convertedType === "TIME_MICROS"
  ) {
    return { kind: "time", unit: column.logical_type?.unit ?? convertedType };
  }
  return null;
}

function parquetDtype(column: ParquetSchemaColumn): string {
  if (column.logical_type?.type === "STRING" || column.converted_type === "UTF8") {
    return "Utf8";
  }

  const timeInfo = temporalInfo(column);
  if (timeInfo?.kind === "timestamp") {
    const unit = String(timeInfo.unit ?? "MICROS").toUpperCase();
    const timeUnit = unit.includes("NANOS") ? "ns" : unit.includes("MILLIS") ? "ms" : "us";
    return `Datetime(time_unit='${timeUnit}', time_zone=None)`;
  }
  if (timeInfo?.kind === "date") {
    return "Date";
  }
  if (timeInfo?.kind === "time") {
    return "Time";
  }

  switch (column.type) {
    case "BOOLEAN":
      return "Boolean";
    case "DOUBLE":
      return "Float64";
    case "FLOAT":
      return "Float32";
    case "INT64":
      return "Int64";
    case "INT32":
      return "Int32";
    case "INT16":
      return "Int16";
    case "INT8":
      return "Int8";
    case "BYTE_ARRAY":
      return "Binary";
    default:
      return column.type ?? "Unknown";
  }
}

function timestampToDate(value: bigint | number, unit: string | undefined): Date {
  const resolved = String(unit ?? "MICROS").toUpperCase();

  if (typeof value === "bigint") {
    if (resolved.includes("NANOS")) return new Date(Number(value / BigInt(1_000_000)));
    if (resolved.includes("MILLIS")) return new Date(Number(value));
    return new Date(Number(value / BigInt(1_000)));
  }

  if (resolved.includes("NANOS")) return new Date(value / 1_000_000);
  if (resolved.includes("MILLIS")) return new Date(value);
  return new Date(value / 1_000);
}

function coerceTemporalValue(value: unknown, info: TemporalInfo | null): unknown {
  if (value == null || info == null) return value;
  if (value instanceof Date) return value;

  if (info.kind === "timestamp") {
    if (typeof value === "bigint" || typeof value === "number") {
      return timestampToDate(value, info.unit);
    }
    if (typeof value === "string") {
      return new Date(value);
    }
  }

  if (info.kind === "date") {
    if (typeof value === "number") return new Date(value * 86_400_000);
    if (typeof value === "string") return new Date(value);
  }

  return value;
}

function sampleValueToString(value: unknown, info: TemporalInfo | null): string | null {
  if (value == null) return null;

  const coerced = coerceTemporalValue(value, info);
  if (coerced instanceof Date && !Number.isNaN(coerced.getTime())) {
    return coerced.toISOString();
  }
  if (typeof coerced === "bigint") return coerced.toString();
  return String(coerced);
}

function extractDateOnly(value: unknown, info: TemporalInfo | null): string | null {
  if (value == null) return null;

  if (typeof value === "string") {
    const match = value.match(/^(\d{4}-\d{2}-\d{2})/);
    if (match) return match[1];
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) return parsed.toISOString().slice(0, 10);
    return null;
  }

  const coerced = coerceTemporalValue(value, info);
  if (coerced instanceof Date && !Number.isNaN(coerced.getTime())) {
    return coerced.toISOString().slice(0, 10);
  }

  return null;
}

function sampleIndices(total: number, n = 15): number[] {
  if (total <= 0) return [];
  if (total <= n) return Array.from({ length: total }, (_, index) => index);

  const step = (total - 1) / (n - 1);
  return Array.from(new Set(Array.from({ length: n }, (_, index) => Math.round(index * step))));
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

async function deriveSampleRows(
  file: ArrayBuffer,
  metadata: ParquetMetadata,
  temporalColumns: Map<string, TemporalInfo>,
): Promise<Array<Record<string, string | null>>> {
  const total = Number(metadata.num_rows);
  if (total === 0) return [];

  const indices = sampleIndices(total);
  const rows =
    total <= 15
      ? await readRows(file, metadata, 0, total)
      : (
          await Promise.all(
            indices.map(async (index) => {
              const [row] = await readRows(file, metadata, index, index + 1);
              return row;
            }),
          )
        ).filter((row): row is Record<string, unknown> => Boolean(row));

  return rows.map((row) => {
    const formatted: Record<string, string | null> = {};
    for (const [key, value] of Object.entries(row)) {
      formatted[key] = sampleValueToString(value, temporalColumns.get(key) ?? null);
    }
    return formatted;
  });
}

async function deriveDateRange(
  file: ArrayBuffer,
  metadata: ParquetMetadata,
  temporalColumns: Map<string, TemporalInfo>,
): Promise<{ start: string; end: string }> {
  for (const candidate of ["timestamp", "date", "time", "datetime"]) {
    const column = metadata.schema.find((item) => item.name === candidate);
    if (!column) continue;

    const rows = await readRows(file, metadata, 0, Number(metadata.num_rows), [candidate]);
    const dates = rows
      .map((row) => extractDateOnly(row[candidate], temporalColumns.get(candidate) ?? null))
      .filter((value): value is string => value != null);

    if (dates.length > 0) {
      const sorted = [...dates].sort();
      return { start: sorted[0], end: sorted[sorted.length - 1] };
    }
  }

  return { start: "", end: "" };
}

export async function deriveRawDataData(
  payload: RawDataPersistedData,
  parquetBytes: Uint8Array,
): Promise<RawDataData> {
  const file = toArrayBuffer(parquetBytes);
  const { parquetMetadata } = await import("hyparquet");

  const metadata = parquetMetadata(file) as ParquetMetadata;
  const columns = leafColumns(metadata);
  const temporalColumns = new Map(
    columns
      .map((column) => [column.name, temporalInfo(column)] as const)
      .filter((entry): entry is [string, TemporalInfo] => entry[1] != null),
  );
  const descriptionByName = new Map(
    (payload.column_descriptions ?? []).map((column) => [column.name, column.description] as const),
  );

  return {
    n_records: Number(metadata.num_rows),
    n_columns: columns.length,
    date_range: await deriveDateRange(file, metadata, temporalColumns),
    sample: await deriveSampleRows(file, metadata, temporalColumns),
    column_descriptions: columns.map((column) => ({
      name: column.name,
      dtype: parquetDtype(column),
      description: descriptionByName.get(column.name) ?? "",
    })),
  };
}
