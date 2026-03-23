import { NextResponse } from "next/server";
import { readData } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

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

export async function GET(
  request: Request,
  { params }: { params: Promise<{ workspaceId: string; stage: string }> },
) {
  const { workspaceId, stage } = await params;

  const safeStage = stage.trim();
  if (!safeStage || /[\\/]/.test(safeStage)) {
    return NextResponse.json({ error: "Invalid route parameters" }, { status: 400 });
  }
  const workspaceAccess = await requireWorkspaceAccess(request, workspaceId);
  if (!workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const { workspaceId: safeWorkspaceId } = workspaceAccess;

  try {
    const raw = await readData(`${safeWorkspaceId}/run/${safeStage}.json`);

    try {
      return NextResponse.json(parseStoredStagePayload(raw));
    } catch (error) {
      return NextResponse.json(
        {
          error: `Invalid persisted data for ${stage}: ${
            error instanceof Error ? error.message : String(error)
          }`,
        },
        { status: 500 },
      );
    }
  } catch {
    return NextResponse.json({ error: `No data for ${stage}` }, { status: 404 });
  }
}
