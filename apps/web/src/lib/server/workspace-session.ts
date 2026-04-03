import { cookies } from "next/headers";
import { getIronSession, type SessionOptions } from "iron-session";
import "@/lib/server/root-env";
import { deriveAppSecret } from "@/lib/server/app-secret";

const WORKSPACE_SESSION_COOKIE = "workspace_session";
const WORKSPACE_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30;
const MAX_AUTHORIZED_WORKSPACES = 32;

type WorkspaceSessionStore = {
  workspaceIds?: string[];
};

function getSessionOptions(): SessionOptions {
  return {
    password: deriveAppSecret("workspace-session"),
    cookieName: WORKSPACE_SESSION_COOKIE,
    ttl: WORKSPACE_SESSION_MAX_AGE_SECONDS,
    cookieOptions: {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      path: "/",
    },
  };
}

async function getWorkspaceSessionStore() {
  return getIronSession<WorkspaceSessionStore>(await cookies(), getSessionOptions());
}

function normalizeWorkspaceIds(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const normalized = new Set<string>();
  for (const entry of value) {
    if (typeof entry !== "string") {
      continue;
    }
    const trimmed = entry.trim();
    if (trimmed) {
      normalized.add(trimmed);
    }
  }

  return Array.from(normalized).slice(0, MAX_AUTHORIZED_WORKSPACES);
}

export async function readAuthorizedWorkspaceIds(): Promise<string[]> {
  const session = await getWorkspaceSessionStore();
  return normalizeWorkspaceIds(session.workspaceIds);
}

export async function hasWorkspaceSessionAccess(workspaceId: string): Promise<boolean> {
  const workspaceIds = await readAuthorizedWorkspaceIds();
  return workspaceIds.includes(workspaceId);
}

export async function authorizeWorkspaceInSession(workspaceId: string): Promise<void> {
  const session = await getWorkspaceSessionStore();
  const existing = normalizeWorkspaceIds(session.workspaceIds).filter((id) => id !== workspaceId);
  session.workspaceIds = [workspaceId, ...existing].slice(0, MAX_AUTHORIZED_WORKSPACES);
  await session.save();
}

export async function replaceAuthorizedWorkspaceIds(workspaceIds: string[]): Promise<void> {
  const session = await getWorkspaceSessionStore();
  session.workspaceIds = normalizeWorkspaceIds(workspaceIds);
  await session.save();
}

export async function clearAuthorizedWorkspaceIds(): Promise<void> {
  await replaceAuthorizedWorkspaceIds([]);
}
