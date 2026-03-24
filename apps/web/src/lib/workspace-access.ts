import { createHash, timingSafeEqual } from "node:crypto";
import { NextResponse } from "next/server";
import { readData, writeData } from "@/lib/storage";

const MAX_WORKSPACE_ID_LENGTH = 200;
const ACCESS_FILE_VERSION = 1;
const ACCESS_COOKIE_PREFIX = "workspace-access-";
const ACCESS_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;

type WorkspaceAccessRecord = {
  version: number;
  codeHash: string;
  createdAt: string;
};

export type WorkspaceAccessDecision =
  | { ok: true; setCookieCode?: string }
  | { ok: false; response: NextResponse };

export type WorkspaceAccessOptions = {
  allowCreate?: boolean;
};

export type WorkspaceAccessRequirement =
  | { ok: true; workspaceId: string; setCookieCode?: string }
  | { ok: false; response: NextResponse };

export function normalizeWorkspaceId(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > MAX_WORKSPACE_ID_LENGTH) {
    return null;
  }

  if (!/^[A-Za-z0-9_-]+$/.test(trimmed)) {
    return null;
  }

  return trimmed;
}

export function hashWorkspaceAccessCode(workspaceId: string, accessCode: string): string {
  return createHash("sha256")
    .update(`workspace-access:${workspaceId}:${accessCode}`)
    .digest("hex");
}

export function getWorkspaceAccessCookieName(workspaceId: string): string {
  const suffix = createHash("sha256").update(workspaceId).digest("hex").slice(0, 24);
  return `${ACCESS_COOKIE_PREFIX}${suffix}`;
}

function getAccessPath(workspaceId: string): string {
  return `${workspaceId}/access.json`;
}

function parseCookieHeader(cookieHeader: string | null): Map<string, string> {
  const cookies = new Map<string, string>();
  if (!cookieHeader) {
    return cookies;
  }

  for (const chunk of cookieHeader.split(";")) {
    const [rawName, ...rawValue] = chunk.trim().split("=");
    if (!rawName || rawValue.length === 0) {
      continue;
    }
    cookies.set(rawName, decodeURIComponent(rawValue.join("=")));
  }

  return cookies;
}

function constantTimeEquals(left: string, right: string): boolean {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && timingSafeEqual(leftBuffer, rightBuffer);
}

async function readWorkspaceAccessRecord(workspaceId: string): Promise<WorkspaceAccessRecord | null> {
  try {
    return JSON.parse(await readData(getAccessPath(workspaceId))) as WorkspaceAccessRecord;
  } catch {
    return null;
  }
}

export async function writeWorkspaceAccessRecord(workspaceId: string, accessCode: string): Promise<void> {
  const record: WorkspaceAccessRecord = {
    version: ACCESS_FILE_VERSION,
    codeHash: hashWorkspaceAccessCode(workspaceId, accessCode),
    createdAt: new Date().toISOString(),
  };
  await writeData(getAccessPath(workspaceId), JSON.stringify(record, null, 2));
}

async function verifyWorkspaceAccessCode(workspaceId: string, accessCode: string): Promise<boolean> {
  const record = await readWorkspaceAccessRecord(workspaceId);
  if (!record) {
    return false;
  }
  return constantTimeEquals(record.codeHash, hashWorkspaceAccessCode(workspaceId, accessCode));
}

function readWorkspaceAccessCookie(request: Request, workspaceId: string): string | null {
  return parseCookieHeader(request.headers.get("cookie")).get(getWorkspaceAccessCookieName(workspaceId)) ?? null;
}

export async function authorizeWorkspaceRequest(
  request: Request,
  workspaceId: string,
  accessCode?: string | null,
  options: WorkspaceAccessOptions = {},
): Promise<WorkspaceAccessDecision> {
  const { allowCreate = false } = options;
  const record = await readWorkspaceAccessRecord(workspaceId);

  if (!record) {
    if (accessCode && allowCreate) {
      await writeWorkspaceAccessRecord(workspaceId, accessCode);
      return { ok: true, setCookieCode: accessCode };
    }

    if (!accessCode) {
      return {
        ok: false,
        response: NextResponse.json({ error: "Workspace access required" }, { status: 401 }),
      };
    }

    return {
      ok: false,
      response: NextResponse.json({ error: "Workspace access denied" }, { status: 403 }),
    };
  }

  const cookieCode = readWorkspaceAccessCookie(request, workspaceId);
  if (cookieCode && constantTimeEquals(record.codeHash, hashWorkspaceAccessCode(workspaceId, cookieCode))) {
    return { ok: true };
  }

  if (accessCode && (await verifyWorkspaceAccessCode(workspaceId, accessCode))) {
    return { ok: true, setCookieCode: accessCode };
  }

  return {
    ok: false,
    response: NextResponse.json({ error: "Workspace access denied" }, { status: 403 }),
  };
}

export async function requireWorkspaceAccess(
  request: Request,
  rawWorkspaceId: string | null | undefined,
  {
    accessCode,
    allowCreate,
  }: WorkspaceAccessOptions & { accessCode?: string | null } = {},
): Promise<WorkspaceAccessRequirement> {
  const workspaceId =
    typeof rawWorkspaceId === "string" ? normalizeWorkspaceId(rawWorkspaceId) : null;
  if (!workspaceId) {
    return {
      ok: false,
      response: NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 }),
    };
  }

  const authorization = await authorizeWorkspaceRequest(request, workspaceId, accessCode, {
    allowCreate,
  });
  if (!authorization.ok) {
    return {
      ok: false,
      response: authorization.response,
    };
  }

  return {
    ok: true,
    workspaceId,
    setCookieCode: authorization.setCookieCode,
  };
}

export function setWorkspaceAccessCookie(
  response: NextResponse,
  workspaceId: string,
  accessCode: string,
): NextResponse {
  response.cookies.set(getWorkspaceAccessCookieName(workspaceId), accessCode, {
    path: "/",
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: ACCESS_COOKIE_MAX_AGE_SECONDS,
  });
  return response;
}
