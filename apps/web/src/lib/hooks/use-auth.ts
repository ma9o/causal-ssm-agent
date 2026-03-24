"use client";

import { useCallback, useEffect, useState } from "react";
import type { AccessStatus } from "@/lib/auth-status";
import { getIdentity, setIdentity, type WorkspaceIdentity } from "@/lib/identity";
import { generateAnonymousWorkspaceId } from "@/lib/workspace-id";
import { generateWorkspaceAccessCode } from "@/lib/resume-key";

export type AuthState = {
  /** Server-derived OpenRouter access status */
  access: AccessStatus | null;
  /** True when user has no access at all (no key + no trial credits) */
  noAccess: boolean;
  /** Sign out — clears the server session but preserves identity */
  signOut: () => Promise<void>;
  /** Ensure identity exists; creates anonymous one if needed. Returns workspace identity. */
  ensureIdentity: () => WorkspaceIdentity;
};

export function useAuth(): AuthState {
  const [access, setAccess] = useState<AccessStatus | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch("/api/auth/status", { cache: "no-store" });
      if (!response.ok) {
        throw new Error("Failed to load auth status");
      }
      setAccess((await response.json()) as AccessStatus);
    } catch {
      setAccess(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const signOut = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
    } finally {
      await refresh();
    }
  }, [refresh]);

  const ensureIdentity = useCallback((): WorkspaceIdentity => {
    const existing = getIdentity();
    if (existing) {
      return existing;
    }
    const id: WorkspaceIdentity = {
      workspaceId: generateAnonymousWorkspaceId(),
      accessCode: generateWorkspaceAccessCode(),
      kind: "anonymous",
    };
    setIdentity(id);
    return id;
  }, []);

  return {
    access,
    noAccess: access?.canRun === false,
    signOut,
    ensureIdentity,
  };
}
