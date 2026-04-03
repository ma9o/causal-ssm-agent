import { NextResponse } from "next/server";
import { isSharedWorkspaceId } from "@/lib/shared-workspaces";
import { authorizeWorkspaceInSession, hasWorkspaceSessionAccess } from "@/lib/server/workspace-session";
import {
  authorizeWorkspaceForOpenRouterUser,
  hasOpenRouterWorkspaceAccess,
  resolveWorkspaceOwnershipContext,
  type WorkspaceOwnershipContext,
} from "@/lib/server/workspace-ownership";
import { prefixExists } from "@/lib/storage";

const MAX_WORKSPACE_ID_LENGTH = 200;

export type WorkspaceAccessDecision =
  | { ok: true }
  | { ok: false; response: NextResponse };

export type WorkspaceAccessOptions = {
  allowCreate?: boolean;
};

export type WorkspaceAccessRequirement =
  | { ok: true; workspaceId: string; creationPending: boolean }
  | { ok: false; response: NextResponse };

type AuthorizedWorkspaceRequestResult =
  | { ok: true; workspaceId: string; creationPending: boolean }
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

function deny403(): WorkspaceAccessDecision {
  return {
    ok: false,
    response: NextResponse.json({ error: "Workspace access denied" }, { status: 403 }),
  };
}

async function hasExistingAccess(
  ownership: WorkspaceOwnershipContext,
  workspaceId: string,
): Promise<boolean> {
  if (ownership.mode === "local") {
    return prefixExists(`${workspaceId}/`);
  }
  if (ownership.mode === "user") {
    return hasOpenRouterWorkspaceAccess(ownership.userId, workspaceId);
  }
  return hasWorkspaceSessionAccess(workspaceId);
}

async function finalizeNewAccess(
  ownership: WorkspaceOwnershipContext,
  workspaceId: string,
): Promise<void> {
  if (ownership.mode === "user") {
    await authorizeWorkspaceForOpenRouterUser(ownership.userId, workspaceId);
    return;
  }

  if (ownership.mode === "anonymous") {
    await authorizeWorkspaceInSession(workspaceId);
  }
}

export async function authorizeWorkspaceRequest(
  _request: Request,
  workspaceId: string,
  options: WorkspaceAccessOptions = {},
): Promise<AuthorizedWorkspaceRequestResult> {
  const { allowCreate = false } = options;

  if (isSharedWorkspaceId(workspaceId)) {
    return allowCreate ? deny403() : { ok: true, workspaceId, creationPending: false };
  }

  const ownership = await resolveWorkspaceOwnershipContext();

  if (await hasExistingAccess(ownership, workspaceId)) {
    return { ok: true, workspaceId, creationPending: false };
  }

  if (!allowCreate) {
    if (ownership.mode === "anonymous") {
      return {
        ok: false,
        response: NextResponse.json({ error: "Workspace access required" }, { status: 401 }),
      };
    }
    return deny403();
  }

  if (await prefixExists(`${workspaceId}/`)) {
    return deny403();
  }

  return { ok: true, workspaceId, creationPending: true };
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

  let authorization: AuthorizedWorkspaceRequestResult;
  try {
    authorization = await authorizeWorkspaceRequest(request, workspaceId, { allowCreate });
  } catch {
    return {
      ok: false,
      response: NextResponse.json({ error: "Workspace session is unavailable" }, { status: 500 }),
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
    creationPending: authorization.creationPending,
  };
}

export async function finalizeWorkspaceCreate(workspaceId: string): Promise<void> {
  const normalizedWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!normalizedWorkspaceId || isSharedWorkspaceId(normalizedWorkspaceId)) {
    return;
  }

  await finalizeNewAccess(await resolveWorkspaceOwnershipContext(), normalizedWorkspaceId);
}
