import { NextResponse } from "next/server";
import { isSharedWorkspaceId } from "@/lib/shared-workspaces";
import { authorizeWorkspaceInSession, hasWorkspaceSessionAccess } from "@/lib/server/workspace-session";
import { prefixExists } from "@/lib/storage";

const MAX_WORKSPACE_ID_LENGTH = 200;

export type WorkspaceAccessDecision =
  | { ok: true }
  | { ok: false; response: NextResponse };

export type WorkspaceAccessOptions = {
  allowCreate?: boolean;
};

export type WorkspaceAccessRequirement =
  | { ok: true; workspaceId: string }
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

export async function authorizeWorkspaceRequest(
  _request: Request,
  workspaceId: string,
  options: WorkspaceAccessOptions = {},
): Promise<WorkspaceAccessDecision> {
  const { allowCreate = false } = options;

  if (isSharedWorkspaceId(workspaceId)) {
    return { ok: true };
  }

  if (await hasWorkspaceSessionAccess(workspaceId)) {
    return { ok: true };
  }

  if (!allowCreate) {
    return {
      ok: false,
      response: NextResponse.json({ error: "Workspace access required" }, { status: 401 }),
    };
  }

  if (await prefixExists(`${workspaceId}/`)) {
    return {
      ok: false,
      response: NextResponse.json({ error: "Workspace access denied" }, { status: 403 }),
    };
  }

  await authorizeWorkspaceInSession(workspaceId);
  return { ok: true };
}

function workspaceSessionErrorResponse(): NextResponse {
  return NextResponse.json(
    { error: "Workspace session is unavailable" },
    { status: 500 },
  );
}

export async function requireWorkspaceAccess(
  request: Request,
  rawWorkspaceId: string | null | undefined,
  { allowCreate }: WorkspaceAccessOptions = {},
): Promise<WorkspaceAccessRequirement> {
  const workspaceId =
    typeof rawWorkspaceId === "string" ? normalizeWorkspaceId(rawWorkspaceId) : null;
  if (!workspaceId) {
    return {
      ok: false,
      response: NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 }),
    };
  }

  let authorization: WorkspaceAccessDecision;
  try {
    authorization = await authorizeWorkspaceRequest(request, workspaceId, { allowCreate });
  } catch {
    return {
      ok: false,
      response: workspaceSessionErrorResponse(),
    };
  }

  if (!authorization.ok) {
    return {
      ok: false,
      response: authorization.response,
    };
  }

  return {
    ok: true,
    workspaceId,
  };
}
