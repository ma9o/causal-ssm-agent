import { readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { NextResponse } from "next/server";

const RESULTS_DIR = resolve(process.cwd(), "..", "data-pipeline", "results");
const FIXTURES_DIR = resolve(process.cwd(), "test", "fixtures");

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string; stage: string }> },
) {
  const { runId, stage } = await params;
  const mockFixture = process.env.NEXT_PUBLIC_MOCK_DATA;
  const isMock = !!mockFixture && mockFixture !== "false";

  // Sanitize path components to prevent directory traversal
  const safeRunId = basename(runId);
  const safeStage = basename(stage);

  const paths: Array<{ path: string; root: string }> = [
    { path: resolve(join(RESULTS_DIR, safeRunId, `${safeStage}.json`)), root: RESULTS_DIR },
    ...(isMock
      ? [
          {
            path: resolve(join(FIXTURES_DIR, basename(mockFixture), `${safeStage}.json`)),
            root: FIXTURES_DIR,
          },
        ]
      : []),
  ];

  for (const { path: filePath, root } of paths) {
    if (!filePath.startsWith(root)) continue;
    try {
      const data = await readFile(filePath, "utf-8");
      return NextResponse.json(JSON.parse(data));
    } catch {
      // Try next path
    }
  }

  return NextResponse.json({ error: `No data for ${stage}` }, { status: 404 });
}
