import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string; stage: string }> },
) {
  const { runId, stage } = await params;
  const isMock = process.env.NEXT_PUBLIC_MOCK_DATA === "true";

  const paths = [
    join(process.cwd(), "..", "data-pipeline", "results", runId, `${stage}.json`),
    ...(isMock
      ? [join(process.cwd(), "test", "fixtures", `${stage}.json`)]
      : []),
  ];

  for (const filePath of paths) {
    try {
      const data = await readFile(filePath, "utf-8");
      return NextResponse.json(JSON.parse(data));
    } catch {
      // Try next path
    }
  }

  return NextResponse.json({ error: `No data for ${stage}` }, { status: 404 });
}
