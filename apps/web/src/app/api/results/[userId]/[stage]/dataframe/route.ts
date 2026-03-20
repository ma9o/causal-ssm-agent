import { basename } from "node:path";
import { readBinary } from "@/lib/storage";

/**
 * Map stage IDs to their parquet artifact filenames.
 * Order matters — first existing file wins (see run_store.py STAGE*_FILENAMES).
 */
const PARQUET_MAP: Record<string, string[]> = {
  "stage-0": ["stage0-raw-input.parquet"],
  "stage-2": ["stage2-raw-data.parquet"],
};

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ userId: string; stage: string }> },
) {
  const { userId, stage } = await params;
  const safeUserId = basename(userId);
  const safeStage = basename(stage);

  const filenames = PARQUET_MAP[safeStage];
  if (!filenames) {
    return new Response("No dataframe available for this stage", { status: 404 });
  }

  for (const filename of filenames) {
    try {
      const bytes = await readBinary(`${safeUserId}/run/${filename}`);
      return new Response(bytes.buffer as ArrayBuffer, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition": `attachment; filename="${filename}"`,
          "Cache-Control": "private, max-age=3600",
        },
      });
    } catch {
      // Try next filename
    }
  }

  return new Response("Parquet file not found", { status: 404 });
}
