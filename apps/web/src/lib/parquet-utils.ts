import type { FileMetaData } from "hyparquet";

export type ParquetSchemaColumn = {
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

export type ParquetMetadata = {
  num_rows: bigint | number;
  schema: ParquetSchemaColumn[];
};

export function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

export async function readParquetRows(
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

export function normalizeParquetScalar(value: unknown): number | boolean | string | null {
  if (value == null) return null;
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "bigint") {
    return value <= BigInt(Number.MAX_SAFE_INTEGER) && value >= BigInt(Number.MIN_SAFE_INTEGER)
      ? Number(value)
      : value.toString();
  }
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "string") {
    return value;
  }
  return String(value);
}
